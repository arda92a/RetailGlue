import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import logging

from retailglue.entities import Detection, collect_embeddings
from retailglue.training.losses import NLLLoss
from retailglue.training.metrics import matcher_metrics

logger = logging.getLogger("retailglue")

FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")
torch.backends.cudnn.deterministic = True

AMP_CUSTOM_FWD_F32 = (
    torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    if hasattr(torch.amp, "custom_fwd")
    else torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
)


@AMP_CUSTOM_FWD_F32
def normalize_keypoints(kpts, size=None):
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    return (kpts - shift[..., None, :]) / scale[..., None, None]


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs, t):
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M, dim, F_dim=None, gamma=1.0):
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=gamma ** -2)

    def forward(self, x):
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    """Per-token confidence predictor for intermediate layer supervision."""

    def __init__(self, dim):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0, desc1):
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        correct1 = la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash=False):
        super().__init__()
        self.enable_flash = allow_flash and FLASH_AVAILABLE
        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, mask=None):
        if self.enable_flash and q.device.type == "cuda":
            args = [x.half().contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
            return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            return v if mask is None else v.nan_to_num()
        s = q.shape[-1] ** -0.5
        sim = torch.einsum("...id,...jd->...ij", q, k) * s
        if mask is not None:
            sim.masked_fill(~mask, -float("inf"))
        attn = F.softmax(sim, -1)
        return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, flash=False, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, x, encoding, mask=None):
        qkv = self.Wqkv(x).unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, flash=False, bias=True):
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        inner_dim = dim_head * num_heads
        self.scale = dim_head ** -0.5
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.flash = Attention(True) if flash and FLASH_AVAILABLE else None

    def _map(self, func, x0, x1):
        return func(x0), func(x1)

    def forward(self, x0, x1, mask=None):
        qk0, qk1 = self._map(self.to_qk, x0, x1)
        v0, v1 = self._map(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = [
            t.unflatten(-1, (self.heads, -1)).transpose(1, 2)
            for t in (qk0, qk1, v0, v1)
        ]
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask)
            m1 = self.flash(qk1, qk0, v0, mask.transpose(-1, -2) if mask is not None else None)
        else:
            qk0, qk1 = qk0 * self.scale ** 0.5, qk1 * self.scale ** 0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1).transpose(-2, -1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10, v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = [t.transpose(1, 2).flatten(start_dim=-2) for t in (m0, m1)]
        m0, m1 = self._map(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.self_attn = SelfBlock(*args, **kwargs)
        self.cross_attn = CrossBlock(*args, **kwargs)

    def forward(self, desc0, desc1, encoding0, encoding1, mask0=None, mask1=None):
        if mask0 is not None and mask1 is not None:
            mask = mask0 & mask1.transpose(-1, -2)
            mask0 = mask0 & mask0.transpose(-1, -2)
            mask1 = mask1 & mask1.transpose(-1, -2)
            desc0 = self.self_attn(desc0, encoding0, mask0)
            desc1 = self.self_attn(desc1, encoding1, mask1)
            return self.cross_attn(desc0, desc1, mask)
        desc0 = self.self_attn(desc0, encoding0)
        desc1 = self.self_attn(desc1, encoding1)
        return self.cross_attn(desc0, desc1)


def sigmoid_log_double_softmax(sim, z0, z1):
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0, desc1):
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        d = mdesc0.shape[-1]
        mdesc0, mdesc1 = mdesc0 / d ** 0.25, mdesc1 / d ** 0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        return sigmoid_log_double_softmax(sim, z0, z1), sim


def filter_matches(scores, th):
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    return (
        torch.where(valid0, m0, -1),
        torch.where(valid1, m1, -1),
        mscores0,
        mscores1,
    )


class LightGlue(nn.Module):
    default_conf = {
        "input_dim": 256,
        "descriptor_dim": 256,
        "n_layers": 9,
        "num_heads": 4,
        "flash": False,
        "filter_threshold": 0.0,
        "checkpointed": False,
        "loss": {
            "gamma": 1.0,
            "nll_balancing": 0.5,
        },
    }

    def __init__(self, conf=None):
        super().__init__()
        self.conf = {**self.default_conf, **(conf or {})}
        if self.conf["input_dim"] != self.conf["descriptor_dim"]:
            self.input_proj = nn.Linear(self.conf["input_dim"], self.conf["descriptor_dim"], bias=True)
        else:
            self.input_proj = nn.Identity()
        head_dim = self.conf["descriptor_dim"] // self.conf["num_heads"]
        self.posenc = LearnableFourierPositionalEncoding(2, head_dim, head_dim)
        h, n, d = self.conf["num_heads"], self.conf["n_layers"], self.conf["descriptor_dim"]
        self.transformers = nn.ModuleList([TransformerLayer(d, h, self.conf["flash"]) for _ in range(n)])
        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList([TokenConfidence(d) for _ in range(n - 1)])

        loss_conf = self.conf.get("loss", {})
        self.loss_fn = NLLLoss(
            nll_balancing=loss_conf.get("nll_balancing", 0.5),
            gamma=loss_conf.get("gamma", 0.0),
        )

    def forward(self, data):
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape

        size0 = data["view0"].get("image_size") if "view0" in data else None
        size1 = data["view1"].get("image_size") if "view1" in data else None
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        desc0 = self.input_proj(data["descriptors0"].contiguous())
        desc1 = self.input_proj(data["descriptors1"].contiguous())
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)

        all_desc0, all_desc1 = [], []
        for i in range(self.conf["n_layers"]):
            if self.conf.get("checkpointed", False) and self.training:
                desc0, desc1 = torch.utils.checkpoint.checkpoint(
                    self.transformers[i], desc0, desc1, encoding0, encoding1,
                    use_reentrant=False,
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1)
            if self.training or i == self.conf["n_layers"] - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[-1](desc0, desc1)
        m0, m1, ms0, ms1 = filter_matches(scores, self.conf["filter_threshold"])

        pred = {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": ms0,
            "matching_scores1": ms1,
            "log_assignment": scores,
        }
        if self.training:
            pred["ref_descriptors0"] = torch.stack(all_desc0, 1)
            pred["ref_descriptors1"] = torch.stack(all_desc1, 1)

        return pred

    def loss(self, pred, data):
        """Compute multi-layer training loss with confidence supervision.

        Returns:
            Tuple of (losses_dict, metrics_dict).
        """
        def loss_params(i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {"log_assignment": la}

        N = pred["ref_descriptors0"].shape[1]
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(-1), data)
        losses = {"total": nll, "last": nll.clone().detach(), **loss_metrics}

        sum_weights = 1.0
        if self.training:
            losses["confidence"] = 0.0

        loss_gamma = self.conf.get("loss", {}).get("gamma", 1.0)

        for i in range(N - 1):
            params_i = loss_params(i)
            nll_i, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

            weight = loss_gamma ** (N - i - 1) if loss_gamma > 0 else i + 1
            sum_weights += weight
            losses["total"] = losses["total"] + nll_i * weight

            if self.training:
                losses["confidence"] += self.token_confidence[i].loss(
                    pred["ref_descriptors0"][:, i],
                    pred["ref_descriptors1"][:, i],
                    params_i["log_assignment"],
                    pred["log_assignment"],
                ) / (N - 1)

        losses["total"] /= sum_weights
        if self.training:
            losses["total"] = losses["total"] + losses["confidence"]

        metrics = {} if self.training else matcher_metrics(pred, data)
        return losses, metrics


class LightGlueMatcher:
    def __init__(self, device='cpu', ransac_threshold=5.0, min_matches=4,
                 weights_path=None, model_config=None):
        self.device = device
        self.ransac_threshold = ransac_threshold
        self.min_matches = min_matches
        self.last_matches_info = {}

        if model_config is None:
            model_config = {
                'input_dim': 128, 'descriptor_dim': 256,
                'n_layers': 9, 'num_heads': 4,
                'flash': False, 'filter_threshold': 0.1,
                'depth_confidence': -1, 'width_confidence': -1,
            }

        self.lightglue = LightGlue(model_config).to(device)
        if weights_path and os.path.isfile(weights_path):
            checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            state_dict = checkpoint.get('model', checkpoint)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_state_dict[k.replace('matcher.', '')] = v
            if not new_state_dict:
                new_state_dict = state_dict
            self.lightglue.load_state_dict(new_state_dict, strict=False)
            logger.info(f"LightGlue fine-tuned weights loaded from {weights_path} "
                        f"(input_dim={model_config.get('input_dim', '?')})")
        else:
            logger.warning(f"No LightGlue weights file found at '{weights_path}' – "
                           f"model will use random initialization. "
                           f"Product-level matching requires fine-tuned weights.")
        self.lightglue.eval()

    def _get_products(self, idx, detections):
        if detections is None or idx >= len(detections):
            return []
        return [d for d in detections[idx] if d.name == 'product']

    def _get_keypoints(self, products):
        if not products:
            return np.array([]).reshape(0, 2)
        return np.array([[(p.xmin + p.xmax) / 2, (p.ymin + p.ymax) / 2] for p in products], dtype=np.float32)

    def _determine_image_order(self, products0, products1, matches, w0, w1):
        if not matches:
            return 'left_right'
        avg_x0 = np.mean([(products0[m[0]].xmin + products0[m[0]].xmax) / 2 for m in matches])
        avg_x1 = np.mean([(products1[m[1]].xmin + products1[m[1]].xmax) / 2 for m in matches])
        return 'left_right' if avg_x0 / w0 > avg_x1 / w1 else 'right_left'

    def _is_partial_product(self, product, image_width, image_height, edge_threshold_ratio=0.02):
        edge_threshold_x = image_width * edge_threshold_ratio
        at_left = product.xmin <= edge_threshold_x
        at_right = product.xmax >= image_width - edge_threshold_x
        if at_left and not at_right:
            return True, 'left'
        elif at_right and not at_left:
            return True, 'right'
        return False, 'none'

    def _calculate_aspect_ratio(self, product):
        return (product.xmax - product.xmin) / (product.ymax - product.ymin + 1e-6)

    def _adjust_keypoints_for_partial_products(self, products0, products1, matches,
                                                image_order, w0, h0, w1, h1,
                                                aspect_ratio_threshold=0.6):
        pts0_expanded, pts1_expanded, adjusted_indices = [], [], []
        for match_idx, m in enumerate(matches):
            p0, p1 = products0[m[0]], products1[m[1]]
            is_partial0, side0 = self._is_partial_product(p0, w0, h0)
            is_partial1, side1 = self._is_partial_product(p1, w1, h1)
            ar0, ar1 = self._calculate_aspect_ratio(p0), self._calculate_aspect_ratio(p1)

            cx0, cy0 = (p0.xmin + p0.xmax) / 2, (p0.ymin + p0.ymax) / 2
            cx1, cy1 = (p1.xmin + p1.xmax) / 2, (p1.ymin + p1.ymax) / 2

            pts0_5 = [[cx0, cy0], [p0.xmin, p0.ymin], [p0.xmax, p0.ymin], [p0.xmax, p0.ymax], [p0.xmin, p0.ymax]]
            pts1_5 = [[cx1, cy1], [p1.xmin, p1.ymin], [p1.xmax, p1.ymin], [p1.xmax, p1.ymax], [p1.xmin, p1.ymax]]

            needs_adjustment = False
            if is_partial1 and not is_partial0 and ar1 < ar0 * aspect_ratio_threshold:
                needs_adjustment = True
                visible_ratio = ar1 / ar0 if ar0 > 0 else 1.0
                p0_width = p0.xmax - p0.xmin
                visible_width = p0_width * visible_ratio
                adjust_left = (image_order == 'left_right' and side1 == 'right') or (image_order == 'right_left' and side1 == 'left')
                if adjust_left:
                    new_xmax = p0.xmin + visible_width
                    new_cx = (p0.xmin + new_xmax) / 2
                    pts0_5 = [[new_cx, cy0], [p0.xmin, p0.ymin], [new_xmax, p0.ymin], [new_xmax, p0.ymax], [p0.xmin, p0.ymax]]
                else:
                    new_xmin = p0.xmax - visible_width
                    new_cx = (p0.xmax + new_xmin) / 2
                    pts0_5 = [[new_cx, cy0], [new_xmin, p0.ymin], [p0.xmax, p0.ymin], [p0.xmax, p0.ymax], [new_xmin, p0.ymax]]
            elif is_partial0 and not is_partial1 and ar0 < ar1 * aspect_ratio_threshold:
                needs_adjustment = True
                visible_ratio = ar0 / ar1 if ar1 > 0 else 1.0
                visible_width = (p1.xmax - p1.xmin) * visible_ratio
                adjust_left = (image_order == 'left_right' and side0 == 'right') or (image_order == 'right_left' and side0 == 'left')
                if adjust_left:
                    new_xmax = p1.xmin + visible_width
                    new_cx = (p1.xmin + new_xmax) / 2
                    pts1_5 = [[new_cx, cy1], [p1.xmin, p1.ymin], [new_xmax, p1.ymin], [new_xmax, p1.ymax], [p1.xmin, p1.ymax]]
                else:
                    new_xmin = p1.xmax - visible_width
                    new_cx = (p1.xmax + new_xmin) / 2
                    pts1_5 = [[new_cx, cy1], [new_xmin, p1.ymin], [p1.xmax, p1.ymin], [p1.xmax, p1.ymax], [new_xmin, p1.ymax]]

            pts0_expanded.extend(pts0_5)
            pts1_expanded.extend(pts1_5)
            adjusted_indices.extend([match_idx if needs_adjustment else -1] * 5)

        return pts0_expanded, pts1_expanded, adjusted_indices

    def _match_with_lightglue(self, kpts0, kpts1, descs0, descs1, shape0, shape1):
        if len(kpts0) == 0 or len(kpts1) == 0:
            return []
        h0, w0 = shape0
        h1, w1 = shape1
        kpts0_t = torch.from_numpy(kpts0).float().unsqueeze(0).to(self.device)
        kpts1_t = torch.from_numpy(kpts1).float().unsqueeze(0).to(self.device)
        desc0_t = torch.from_numpy(np.asarray(descs0)).float().unsqueeze(0).to(self.device)
        desc1_t = torch.from_numpy(np.asarray(descs1)).float().unsqueeze(0).to(self.device)

        data = {
            'keypoints0': kpts0_t, 'keypoints1': kpts1_t,
            'descriptors0': desc0_t, 'descriptors1': desc1_t,
            'view0': {'image_size': torch.tensor([[w0, h0]], dtype=torch.float32).to(self.device)},
            'view1': {'image_size': torch.tensor([[w1, h1]], dtype=torch.float32).to(self.device)},
        }
        with torch.no_grad():
            pred = self.lightglue(data)

        matches = pred['matches0'][0].cpu().numpy()
        scores = pred.get('matching_scores0', [None])[0]
        if scores is not None:
            scores = scores.cpu().numpy()

        result = []
        for i, m in enumerate(matches):
            if m >= 0:
                score = float(scores[i]) if scores is not None else 1.0
                result.append((i, int(m), score))
        return result

    def _process_pair(self, i0, i1, pair_idx, detections=None, skip_min_matches_filter=False):
        image0, idx0 = (i0.image, i0.idx) if hasattr(i0, 'image') else (i0, pair_idx)
        image1, idx1 = (i1.image, i1.idx) if hasattr(i1, 'image') else (i1, pair_idx + 1)

        products0 = self._get_products(idx0, detections)
        products1 = self._get_products(idx1, detections)

        empty = np.array([], dtype=np.float32).reshape(0, 2)
        if not products0 or not products1:
            self.last_matches_info = {}
            return empty, empty

        embeddings0 = np.array(collect_embeddings(products0), dtype=np.float32)
        embeddings1 = np.array(collect_embeddings(products1), dtype=np.float32)
        kpts0 = self._get_keypoints(products0)
        kpts1 = self._get_keypoints(products1)

        matches = self._match_with_lightglue(kpts0, kpts1, embeddings0, embeddings1, image0.shape[:2], image1.shape[:2])

        if len(matches) < self.min_matches and not skip_min_matches_filter:
            self.last_matches_info = {'products0': products0, 'products1': products1, 'matches': matches}
            return empty, empty

        h0, w0 = image0.shape[:2]
        h1, w1 = image1.shape[:2]
        image_order = self._determine_image_order(products0, products1, matches, w0, w1)

        pts0_exp, pts1_exp, adj_indices = self._adjust_keypoints_for_partial_products(
            products0, products1, matches, image_order, w0, h0, w1, h1)
        match_indices = [idx if idx >= 0 else (i // 5) for i, idx in enumerate(adj_indices)]
        pts0_all = np.array(pts0_exp, dtype=np.float32)
        pts1_all = np.array(pts1_exp, dtype=np.float32)

        self.last_matches_info = {
            'products0': products0, 'products1': products1, 'matches': matches,
            'inlier_matches': matches, 'num_inliers': len(pts0_all), 'image_order': image_order,
        }

        if len(pts0_all) >= 4:
            H, mask = cv2.findHomography(pts0_all, pts1_all, cv2.USAC_MAGSAC,
                                          ransacReprojThreshold=self.ransac_threshold,
                                          confidence=0.999, maxIters=5000)
            if H is not None and mask is not None:
                mask = mask.ravel().astype(bool)
                pts0_in, pts1_in = pts0_all[mask], pts1_all[mask]
                inlier_idx = {match_indices[i] for i, m in enumerate(mask) if m}
                outlier_idx = {match_indices[i] for i, m in enumerate(mask) if not m} - inlier_idx
                self.last_matches_info.update({
                    'inlier_matches': [matches[i] for i in range(len(matches)) if i in inlier_idx],
                    'outlier_matches': [matches[i] for i in range(len(matches)) if i in outlier_idx],
                    'num_inliers': len(pts0_in),
                    'num_outliers': len(pts0_all) - len(pts0_in),
                    'inlier_mask': mask,
                    'pts0_all': pts0_all, 'pts1_all': pts1_all,
                })
                return pts0_in, pts1_in

        return pts0_all, pts1_all

    def inference(self, image_pairs, detections=None, skip_min_matches_filter=False):
        batch_src, batch_tgt = [], []
        for pair_idx, (i0, i1) in enumerate(image_pairs):
            pts0, pts1 = self._process_pair(i0, i1, pair_idx, detections, skip_min_matches_filter)
            batch_src.append(pts0)
            batch_tgt.append(pts1)
        return batch_src, batch_tgt
