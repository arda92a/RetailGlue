from types import SimpleNamespace

from retailglue.config import resolve_path
from retailglue.matchers.lightglue import LightGlueMatcher
from retailglue.matchers.lightgluestick import LightGlueStickMatcher
from retailglue.matchers.gluestick import GlueStickMatcher
from retailglue.matchers.roma import RomaV2Matcher
from retailglue.matchers.bfmatcher import BFMatcher

DINO_MATCHER_NAMES = (
    'lightglue_dinov3_vits', 'lightglue_dinov3_vitb', 'lightglue_dinov3_vitl',
    'lightglue_dinov3_vith', 'lightglue_dinov2_vits',
)

BF_MATCHER_NAMES = (
    'bfmatcher_dinov3_vits', 'bfmatcher_dinov3_vitb', 'bfmatcher_dinov3_vitl',
)


def _cfg_get(cfg, key, default=None):
    """Get a value from config that may be a dict or SimpleNamespace."""
    if isinstance(cfg, SimpleNamespace):
        return getattr(cfg, key, default)
    return cfg.get(key, default)


def create_matcher(model_name: str, device: str = 'cpu', config=None):
    cfg = config or {}

    if model_name in DINO_MATCHER_NAMES:
        dino_dims = {
            'lightglue_dinov3_vits': 384,
            'lightglue_dinov3_vitb': 768,
            'lightglue_dinov3_vitl': 1024,
            'lightglue_dinov3_vith': 1280,
            'lightglue_dinov2_vits': 384,
        }
        input_dim = dino_dims[model_name]
        dino_model_config = {
            'input_dim': input_dim,
            'descriptor_dim': 256,
            'n_layers': 9,
            'num_heads': 4,
            'flash': False,
            'filter_threshold': 0.1,
            'depth_confidence': -1,
            'width_confidence': -1,
        }
        # Per-variant LightGlue weights: cfg.lightglue_weights can be
        # a SimpleNamespace/dict mapping variant names to paths, or a single path string.
        lightglue_weights_cfg = _cfg_get(cfg, 'lightglue_weights', None)
        if lightglue_weights_cfg is not None and not isinstance(lightglue_weights_cfg, str):
            # Per-variant mapping
            weights_path = _cfg_get(lightglue_weights_cfg, model_name, None)
        else:
            weights_path = lightglue_weights_cfg
        if weights_path:
            weights_path = resolve_path(weights_path)
        return LightGlueMatcher(
            device=device,
            ransac_threshold=_cfg_get(cfg, 'lightglue_ransac_threshold', 5.0),
            min_matches=_cfg_get(cfg, 'lightglue_min_matches', 4),
            model_config=dino_model_config,
            weights_path=weights_path,
        )

    if model_name in BF_MATCHER_NAMES:
        return BFMatcher(
            device=device,
            ransac_threshold=_cfg_get(cfg, 'bfmatcher_ransac_threshold',
                                     _cfg_get(cfg, 'lightglue_ransac_threshold', 5.0)),
            min_matches=_cfg_get(cfg, 'bfmatcher_min_matches',
                                 _cfg_get(cfg, 'lightglue_min_matches', 4)),
            ratio_threshold=_cfg_get(cfg, 'bfmatcher_ratio_threshold', 0.75),
            cross_check=_cfg_get(cfg, 'bfmatcher_cross_check', False),
        )

    if model_name == 'roma_v2':
        return RomaV2Matcher(
            device=device,
            num_samples=_cfg_get(cfg, 'roma_num_samples', 5000),
            ransac_threshold=_cfg_get(cfg, 'roma_ransac_threshold', 5.0),
        )

    if model_name in ('gluestick', 'gluestick_no_lines'):
        return GlueStickMatcher(
            device=device,
            max_keypoints=_cfg_get(cfg, 'gluestick_max_keypoints', 1000),
            max_lines=_cfg_get(cfg, 'gluestick_max_lines', 300),
            use_lines=(model_name != 'gluestick_no_lines'),
            ransac_threshold=_cfg_get(cfg, 'gluestick_ransac_threshold', 5.0),
            max_edge=_cfg_get(cfg, 'gluestick_max_edge', 1024),
        )

    if model_name in ('lightgluestick', 'lightgluestick_no_lines'):
        return LightGlueStickMatcher(
            device=device,
            max_keypoints=_cfg_get(cfg, 'lightgluestick_max_keypoints', 2048),
            max_lines=_cfg_get(cfg, 'lightgluestick_max_lines', 250),
            use_lines=(model_name != 'lightgluestick_no_lines'),
            ransac_threshold=_cfg_get(cfg, 'lightgluestick_ransac_threshold', 5.0),
            max_edge=_cfg_get(cfg, 'lightgluestick_max_edge', 1024),
        )

    from retailglue.matchers.hf_model import HFModel
    return HFModel(model_name, device=device)
