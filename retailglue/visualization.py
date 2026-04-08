import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Union, Tuple
from retailglue.entities import BoundingBox, Polygon, Detection, Point
from retailglue.io import to_pil


class Visualizer:
    def __init__(self, image: Union[np.ndarray, Image.Image]):
        self.image = to_pil(image).copy()
        try:
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except IOError:
            self.font = ImageFont.load_default()

    def put_text(self, label: str, x: int, y: int, font_size: int = 16,
                 color=(0, 0, 0), rotation: int = 0):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()
        if rotation != 0:
            txt_img = Image.new('RGBA', (300, 50), (255, 255, 255, 0))
            txt_draw = ImageDraw.Draw(txt_img)
            txt_draw.text((0, 0), label, fill=color, font=font)
            txt_img = txt_img.rotate(rotation, expand=True)
            self.image.paste(txt_img, (x, y), txt_img)
        else:
            draw = ImageDraw.Draw(self.image)
            draw.text((x, y), label, fill=color, font=font)

    def draw_bbox(self, bbox: BoundingBox, label: str = None,
                  color=(179, 19, 18), line_width: int = 4):
        draw = ImageDraw.Draw(self.image)
        shape = bbox.get_pascal_voc_format()
        draw.rectangle(shape, outline=color, width=line_width)
        if label:
            draw.text((bbox.xmin, bbox.ymin - 15), label, fill=color, font=self.font)
        return self

    def draw_polygon(self, polygon: Polygon, label: str = None,
                     color=(179, 19, 18), line_width: int = 4, background: bool = True):
        draw = ImageDraw.Draw(self.image)
        contour = [p.as_tuple() for p in polygon.points]
        draw.polygon(contour, outline=color, width=line_width)
        if label and polygon.points:
            draw.text((polygon.points[0].x, polygon.points[0].y - 15), label, fill=color, font=self.font)
        return self

    def draw_detections(self, detections: List[Detection]):
        for i, det in enumerate(detections):
            color = _color_from_index(i)
            label = f"{i}:{det.name}({det.score*100:.0f}%)" if hasattr(det, 'score') else det.name
            self.draw_bbox(det, label=label, color=color)
        return self

    def save(self, path: str):
        self.image.save(path)
        return self


def _color_from_index(i: int) -> Tuple[int, int, int]:
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    ]
    return colors[i % len(colors)]
