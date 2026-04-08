import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import copy
import logging
import time

import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageOps

from retailglue.config import get_config, resolve_path
from retailglue.detector import YOLODetector
from retailglue.embeddings import extract_dino_embeddings, DINO_VARIANTS, BF_TO_DINO_VARIANT
from retailglue.stitching.stitcher import ImageStitcher, StitchingImage
from retailglue.visualization import Visualizer

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                    force=True)
logger = logging.getLogger("retailglue")
# Ensure all handlers flush immediately
for handler in logging.root.handlers:
    handler.flush()

DEVICE = ('cuda' if torch.cuda.is_available()
          else 'mps' if torch.backends.mps.is_available()
          else 'cpu')
DINO_MODEL_NAMES = tuple(DINO_VARIANTS.keys()) + tuple(BF_TO_DINO_VARIANT.keys())

MODEL_NAMES = [
    'lightglue_dinov3_vits', 'lightglue_dinov3_vitb', 'lightglue_dinov3_vitl', 'lightglue_dinov2_vits',
    'bfmatcher_dinov3_vits', 'bfmatcher_dinov3_vitb', 'bfmatcher_dinov3_vitl',
    'lightgluestick', 'lightgluestick_no_lines',
    'gluestick', 'gluestick_no_lines',
    'roma_v2',
    'superglue', 'lightglue_superpoint', 'lightglue_disk', 'lightglue_minima', 'eloftr',
]

config = get_config()
stitcher_instance = None
sku_yolo_detector = None
global_matches_data = []
global_image_names = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_detector(device):
    global sku_yolo_detector
    weights = (getattr(config.detector, 'sku_yolo_weights', 'weights/sku_yolo/best_sku110k.pt')
               if hasattr(config, 'detector')
               else 'weights/sku_yolo/best_sku110k.pt')
    weights = resolve_path(weights)
    if sku_yolo_detector is None and os.path.exists(weights):
        sku_yolo_detector = YOLODetector(weights, device=device)
    return sku_yolo_detector


def get_stitcher(model_name, device):
    global stitcher_instance
    cfg = copy.copy(config.stitching)
    cfg.model_name = model_name
    cfg.device = device
    cfg.verbose = True          # always verbose for debug outputs
    stitcher_instance = ImageStitcher(config=cfg)
    return stitcher_instance


def read_images(image_file_obj_inputs):
    """Read uploaded image files with EXIF transpose."""
    global global_image_names
    if not image_file_obj_inputs:
        global_image_names = []
        return []
    names = [str(p).replace('\\', '/').split('/')[-1] for p in image_file_obj_inputs]
    global_image_names = names
    images = [np.array(ImageOps.exif_transpose(Image.open(x).convert('RGB')))
              for x in image_file_obj_inputs]
    return [(img, name) for img, name in zip(images, names)]


def _detect_and_embed(images, device, model_name):
    """Run SKU YOLO detection + DINO embedding extraction. Returns detections list."""
    print(f"[PIPELINE] _detect_and_embed: {len(images)} images, model={model_name}, device={device}", flush=True)
    detector = get_detector(device)
    if detector is None:
        return None
    detections = detector.detect(images, conf=0.25)
    logger.info(f"Product detections per image: {[len(d) for d in detections]}")

    if model_name in DINO_MODEL_NAMES:
        dino_variant = BF_TO_DINO_VARIANT.get(model_name, model_name)
        dino_weights = None
        if hasattr(config, 'embeddings') and hasattr(config.embeddings, 'dino_weights'):
            dw = config.embeddings.dino_weights
            dino_weights = getattr(dw, dino_variant, None)
            if dino_weights:
                dino_weights = resolve_path(dino_weights)
        crop_margin = (getattr(config.embeddings, 'crop_margin', 10)
                       if hasattr(config, 'embeddings') else 10)
        extract_dino_embeddings(images, detections, device, variant=model_name,
                                weights_path=dino_weights, crop_margin=crop_margin)
    return detections


# ---------------------------------------------------------------------------
# Visualization helpers  (match the original local_stitching.py)
# ---------------------------------------------------------------------------

def draw_detections_on_image(image, detections):
    """Draw bounding boxes + labels on a single image. Returns annotated copy."""
    vis = Visualizer(image)
    if detections:
        vis.draw_detections(detections)
    return np.array(vis.image)


def draw_matches(image0, image1, left_pts, right_pts, vertical=False,
                 rejected_left_pts=None, rejected_right_pts=None):
    """Draw matched keypoints between two images (green=inliers, red=outliers)."""
    h0, w0 = image0.shape[:2]
    h1, w1 = image1.shape[:2]

    if vertical:
        if w0 > w1:
            image1 = np.pad(image1, ((0, 0), (0, w0 - w1), (0, 0)), mode='constant')
        elif w0 < w1:
            image0 = np.pad(image0, ((0, 0), (0, w1 - w0), (0, 0)), mode='constant')
    else:
        if h0 > h1:
            image1 = np.pad(image1, ((0, h0 - h1), (0, 0), (0, 0)), mode='constant')
        elif h0 < h1:
            image0 = np.pad(image0, ((0, h1 - h0), (0, 0), (0, 0)), mode='constant')

    image_concat = np.concatenate((image0, image1), axis=int(not vertical)).copy()
    offset = np.array([0, h0]) if vertical else np.array([w0, 0])

    red = (255, 0, 0)
    green = (0, 255, 0)

    # Draw outliers (red)
    if (rejected_left_pts is not None and rejected_right_pts is not None
            and len(rejected_left_pts) > 0 and len(rejected_right_pts) > 0):
        rl = rejected_left_pts.astype(int)
        rr = (rejected_right_pts + offset).astype(int)
        for lp, rp in zip(rl, rr):
            cv2.circle(image_concat, tuple(lp), 4, red, -1)
            cv2.circle(image_concat, tuple(rp), 4, red, -1)
            cv2.line(image_concat, tuple(lp), tuple(rp), red, 1, lineType=cv2.LINE_AA)

    # Draw inliers (green)
    if len(left_pts) > 0 and len(right_pts) > 0:
        li = left_pts.astype(int)
        ri = (right_pts + offset).astype(int)
        for lp, rp in zip(li, ri):
            cv2.circle(image_concat, tuple(lp), 4, green, -1)
            cv2.circle(image_concat, tuple(rp), 4, green, -1)
            cv2.line(image_concat, tuple(lp), tuple(rp), green, 1, lineType=cv2.LINE_AA)

    return image_concat


# ---------------------------------------------------------------------------
# Core pipeline functions
# ---------------------------------------------------------------------------

def match_fn(image_file_obj_inputs, device, model_name):
    """Find and visualize keypoint matches between consecutive image pairs."""
    global global_matches_data
    import sys, traceback
    print(f"\n{'='*60}", flush=True)
    print(f"[MATCH] Called with {len(image_file_obj_inputs) if image_file_obj_inputs else 0} images, model={model_name}, device={device}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        return _match_fn_inner(image_file_obj_inputs, device, model_name)
    except Exception as e:
        traceback.print_exc()
        print(f"[MATCH ERROR] {e}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return (gr.update(choices=[], value=None), f"ERROR: {e}", None, [])


def _match_fn_inner(image_file_obj_inputs, device, model_name):
    global global_matches_data

    s = get_stitcher(model_name, device)
    images_data = read_images(image_file_obj_inputs)
    if len(images_data) < 2:
        global_matches_data = []
        return (gr.update(choices=[], value=None),
                "Error: At least 2 images required", None, [])

    images = [img for img, _ in images_data]

    # Detect & embed
    detections = _detect_and_embed(images, device, model_name)

    # Detection overlay gallery
    det_gallery = []
    for i, img in enumerate(images):
        dets = detections[i] if detections else []
        annotated = draw_detections_on_image(img, dets)
        det_count = len(dets)
        det_gallery.append((annotated, f"{global_image_names[i]} ({det_count} products)"))

    all_matches = []
    total_matches, total_rejected = 0, 0

    for i in range(len(images) - 1):
        si1 = StitchingImage(idx=i, image=images[i])
        si2 = StitchingImage(idx=i + 1, image=images[i + 1])

        left_pts, right_pts, rej_left, rej_right = s.get_matching_keypoints(
            [(si1, si2)], detections=detections, skip_min_matches_filter=True)

        left_pts, right_pts = left_pts[0], right_pts[0]
        rej_left = rej_left[0] if len(rej_left) > 0 and len(rej_left[0]) > 0 else None
        rej_right = rej_right[0] if len(rej_right) > 0 and len(rej_right[0]) > 0 else None
        rejected_count = len(rej_left) if rej_left is not None else 0

        match_image = draw_matches(images[i], images[i + 1], left_pts, right_pts,
                                   rejected_left_pts=rej_left, rejected_right_pts=rej_right)
        match_count = len(left_pts)
        total_matches += match_count
        total_rejected += rejected_count

        left_name = global_image_names[i] if i < len(global_image_names) else f"Image_{i + 1}"
        right_name = (global_image_names[i + 1]
                      if (i + 1) < len(global_image_names) else f"Image_{i + 2}")
        pair_text = f"{left_name} <-> {right_name}"
        if rejected_count > 0:
            pair_text += f" ({rejected_count} rejected)"

        all_matches.append({
            'image': match_image,
            'count': match_count,
            'rejected': rejected_count,
            'inlier_count': match_count,
            'pair': pair_text,
        })

    global_matches_data = all_matches

    info_msg = f"Total matches: {total_matches}"
    if total_rejected > 0:
        info_msg += f", Rejected: {total_rejected} (red)"

    if not all_matches:
        return (gr.update(choices=[], value=None), "Matching failed", None, det_gallery)

    choices = [f"{m['pair']} ({m['count']} match, {m['inlier_count']} inliers)"
               for m in all_matches]
    return (gr.update(choices=choices, value=choices[0], interactive=True),
            info_msg, all_matches[0]['image'], det_gallery)


def update_selected_match(selected_choice):
    global global_matches_data
    if not global_matches_data or selected_choice is None:
        return None
    for m in global_matches_data:
        label = f"{m['pair']} ({m['count']} match, {m['inlier_count']} inliers)"
        if label == selected_choice:
            return m['image']
    return None


def stitch_fn(image_file_obj_inputs, device, model_name, draw_products):
    """Full stitching pipeline with graph visualizations and detection drawing."""
    import sys, traceback
    print(f"\n{'='*60}", flush=True)
    print(f"[STITCH] Called with {len(image_file_obj_inputs) if image_file_obj_inputs else 0} images, model={model_name}, device={device}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        return _stitch_fn_inner(image_file_obj_inputs, device, model_name, draw_products)
    except Exception as e:
        traceback.print_exc()
        print(f"[STITCH ERROR] {e}", flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        return [], f"ERROR: {e}", [], []


def _stitch_fn_inner(image_file_obj_inputs, device, model_name, draw_products):
    s = get_stitcher(model_name, device)
    images_data = read_images(image_file_obj_inputs)
    if not images_data:
        return [], "No images provided", [], []

    images = [img for img, _ in images_data]
    tic = time.time()

    # Detect & embed
    detections = _detect_and_embed(images, device, model_name)

    # Detection overlay gallery
    det_gallery = []
    for i, img in enumerate(images):
        dets = detections[i] if detections else []
        annotated = draw_detections_on_image(img, dets)
        det_count = len(dets)
        det_gallery.append((annotated, f"{global_image_names[i]} ({det_count} products)"))

    # Stitch (verbose=True → returns graph vizs)
    result = s.stitch_images(images, detections=detections)

    # Unpack result based on verbose return format
    if isinstance(result, tuple):
        if detections is not None:
            # With detections: (panoramas, det_results, [graphs]) or (panoramas, det_results)
            if len(result) == 3:
                panoramas, det_results, graph_viz_list = result
            else:
                panoramas, det_results = result[:2]
                graph_viz_list = []
        else:
            # Without detections: (panoramas, [graphs]) or just panoramas
            if len(result) == 2 and isinstance(result[1], list):
                panoramas, graph_viz_list = result
            else:
                panoramas = result if isinstance(result, list) else [result]
                graph_viz_list = []
            det_results = None
    else:
        panoramas = result if isinstance(result, list) else [result]
        graph_viz_list = []
        det_results = None

    # Draw product detections on stitched panoramas
    visualized_panoramas = []
    for pano_idx, pano in enumerate(panoramas):
        if draw_products and det_results is not None and pano_idx < len(det_results):
            vis = Visualizer(pano)
            products = [d for d in det_results[pano_idx] if d.name == 'product']
            if products:
                vis.draw_detections(products)
            visualized_panoramas.append(np.array(vis.image))
        else:
            visualized_panoramas.append(pano)

    elapsed = time.time() - tic
    det_info = ""
    if detections is not None:
        det_counts = [len(d) for d in detections]
        det_info = f"Products per image: {det_counts} | "
    info = (f"{det_info}Stitched {len(images)} images → "
            f"{len(panoramas)} panorama(s) in {elapsed:.2f}s")
    logger.info(info)

    pano_gallery = [Image.fromarray(p) for p in visualized_panoramas]

    # Graph visualizations
    graph_gallery = []
    if graph_viz_list:
        titles = ['Input Graph', 'Filtered Graph', 'Subgraphs']
        for viz, title in zip(graph_viz_list, titles):
            if viz is not None:
                graph_gallery.append((viz, title))

    return pano_gallery, info, graph_gallery, det_gallery


# ---------------------------------------------------------------------------
# Gradio UI  (mirrors local_stitching.py layout)
# ---------------------------------------------------------------------------

def build_interface():
    with gr.Blocks(title="RetailGlue - Retail Shelf Panorama Stitching") as demo:
        gr.Markdown("# RetailGlue: Semantic Product-Level Image Stitching")
        gr.Markdown("Upload shelf images to create panoramic stitched views with "
                     "full debug visualizations.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(label="Upload Images", file_count="multiple",
                                     type="filepath")
                with gr.Row():
                    device_dropdown = gr.Dropdown(choices=["cpu", "cuda", "mps"],
                                                  value=DEVICE, label="Device")
                    model_dropdown = gr.Dropdown(choices=MODEL_NAMES,
                                                 value="lightglue_dinov3_vits",
                                                 label="Matching Model")
                draw_products_checkbox = gr.Checkbox(label="Draw Products on Panorama",
                                                     value=True)
                info_textbox = gr.Textbox(label="Info", interactive=False)
                graph_viz_gallery = gr.Gallery(label="Graph Visualization",
                                               height="auto", columns=3, format="png")

            with gr.Column(scale=2):
                view_output = gr.Gallery(label="Input Images (with detections)",
                                         columns=3, object_fit="contain",
                                         height="auto", format="jpeg")

        # Uploaded image preview
        file_input.change(fn=read_images, inputs=[file_input], outputs=[view_output])

        with gr.Row():
            with gr.Column():
                with gr.Tab("Stitch"):
                    stitch_button = gr.Button("Stitch", variant="primary", size="lg")
                    stitch_output = gr.Gallery(label="Stitched Panoramas", columns=1,
                                               object_fit="contain", height=600,
                                               format="jpeg")

                with gr.Tab("Matches"):
                    with gr.Row():
                        match_button = gr.Button("Match", variant="primary", size="lg")
                        pair_dropdown = gr.Dropdown(choices=[], label="Select Match Pair",
                                                    value=None, interactive=True)
                    match_output = gr.Image(label="Keypoint Matches", height=500)
                    match_det_gallery = gr.Gallery(
                        label="Detections per Image", columns=3,
                        object_fit="contain", height="auto", format="jpeg")

        # Callbacks
        match_button.click(
            fn=match_fn,
            inputs=[file_input, device_dropdown, model_dropdown],
            outputs=[pair_dropdown, info_textbox, match_output, match_det_gallery],
        )
        pair_dropdown.change(
            fn=update_selected_match,
            inputs=[pair_dropdown],
            outputs=[match_output],
        )
        stitch_button.click(
            fn=stitch_fn,
            inputs=[file_input, device_dropdown, model_dropdown, draw_products_checkbox],
            outputs=[stitch_output, info_textbox, graph_viz_gallery, view_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(share=False)
