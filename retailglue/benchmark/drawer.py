import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from retailglue.visualization import Visualizer
from retailglue.entities import Detection, BoundingBox


def save_heatmap_image(TP, FN, TN, FP, path, suffix=""):
    array = [[TP, FN], [FP, TN]]
    df_cm = pd.DataFrame(array, index=["Found", "Not Found"], columns=["Matched", "Unmatched"])
    fig = plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True, fmt='g')
    plt.xlabel("Stitched Image Boxes")
    plt.ylabel("Single Image Boxes")
    plt.savefig(os.path.join(path, f"Heatmap{suffix}.png"))
    plt.close(fig)


def draw_deleted_polygons(image, delete_dict, polygons):
    vis = Visualizer(image)
    for key, values in delete_dict.items():
        for value in values:
            poly = polygons[key][value]
            vis.draw_polygon(poly, label="D", color=(0, 255, 200), background=False)
    return vis


def draw_final_polygons(image, polygons):
    vis = Visualizer(image)
    for poly in polygons:
        vis.draw_polygon(poly, label="S", color=(100, 0, 100), background=False)
    return vis


def draw_unmatched(image, stitched, single):
    vis = Visualizer(image)
    vis.put_text(label="Blue: Stitched(+) | Single(-)", x=20, y=20, font_size=32)
    vis.put_text(label="Green: Stitched(-) | Single(+)", x=20, y=60, font_size=32)
    for st in stitched:
        vis.draw_bbox(st, label="St", color=(0, 0, 255), line_width=8)
    for si in single:
        vis.draw_polygon(si, label="Si", color=(0, 255, 0), line_width=8, background=False)
    return vis


def draw_stitched_image_detections(image, detections):
    vis = Visualizer(image)
    for det in detections:
        vis.draw_bbox(det, label="B", color=(30, 120, 120), line_width=8)
    return vis


def draw_stitched_with_detections(image, detections):
    vis = Visualizer(image)
    det_list = []
    for det in detections:
        if isinstance(det, BoundingBox) and not isinstance(det, Detection):
            det_list.append(Detection(object_id=0, xmin=det.xmin, ymin=det.ymin,
                                      xmax=det.xmax, ymax=det.ymax, score=1.0, name='product'))
        else:
            det_list.append(det)
    vis.draw_detections(det_list)
    return vis
