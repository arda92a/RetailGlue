import math
import cv2
import numpy as np
from typing import Dict, List, Tuple, Union
from shapely.ops import unary_union
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon


class Point:
    def __init__(self, x: int, y: int):
        self.x = int(x)
        self.y = int(y)

    @property
    def geometric(self) -> ShapelyPoint:
        return ShapelyPoint(self.x, self.y)

    @staticmethod
    def mid_point_of(point: 'Point', other_point: 'Point') -> 'Point':
        return Point(x=(point.x + other_point.x) // 2, y=(point.y + other_point.y) // 2)

    def distance_from(self, other: 'Point', method: str = "euclidian") -> int:
        if method == "euclidian":
            return int(((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5)
        return abs(self.x - other.x) + abs(self.y - other.y)

    def within(self, bbox: 'BoundingBox') -> bool:
        return bbox.xmin <= self.x <= bbox.xmax and bbox.ymin <= self.y <= bbox.ymax

    def within_polygon(self, polygon: 'Polygon') -> bool:
        return polygon.geometric.intersects(ShapelyPoint(self.x, self.y))

    def as_list(self) -> List:
        return [self.x, self.y]

    def as_tuple(self) -> Tuple:
        return (self.x, self.y)

    def to_dict(self) -> Dict:
        return {'x': self.x, 'y': self.y}

    @staticmethod
    def from_dict(d) -> 'Point':
        return Point(x=d['x'], y=d['y'])

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"


class Polygon:
    def __init__(self, points: List[Point]):
        if len(points) > 0 and not isinstance(points[0], Point):
            points = [Point(*p) for p in points]
        self.points = points
        self.points = self._sort_ccw()

    @property
    def geometric(self) -> ShapelyPolygon:
        return ShapelyPolygon(self.get_points_tuple())

    def __len__(self):
        return len(self.points)

    @property
    def center(self) -> Point:
        x, y = np.array(self.get_points_list()).mean(axis=0)
        return Point(x=x, y=y)

    @property
    def xcenter(self) -> int:
        return self.center.x

    @property
    def ycenter(self) -> int:
        return self.center.y

    @property
    def sorted_points(self) -> List[Point]:
        return self._sort_ccw()

    @property
    def rect(self) -> 'BoundingBox':
        contours = np.float32(self.get_points_list())
        xmin, ymin, w, h = cv2.boundingRect(contours)
        return BoundingBox(xmin=xmin, ymin=ymin, xmax=xmin + w - 1, ymax=ymin + h - 1)

    def reduce_points(self) -> 'Polygon':
        if len(self.points) == 4:
            return self
        contour = np.expand_dims(np.float32(self.get_points_list()), axis=1)
        reshaped = contour.reshape(-1, 2)
        min_x_pts = reshaped[reshaped[:, 0].argsort()][:2]
        max_x_pts = reshaped[reshaped[:, 0].argsort()][::-1][:2]
        new_points = [
            min_x_pts[min_x_pts[:, 1].argmin()],
            min_x_pts[min_x_pts[:, 1].argmax()],
            max_x_pts[max_x_pts[:, 1].argmax()],
            max_x_pts[max_x_pts[:, 1].argmin()],
        ]
        self.points = [Point(*p) for p in new_points]
        self.points = self._sort_ccw()
        return self

    def _sort_ccw(self) -> List[Point]:
        center = self.center if len(self.points) > 0 else Point(0, 0)
        sorted_pts = list(reversed(sorted(
            self.points,
            key=lambda p: math.atan2(p.y - center.y, p.x - center.x)
        )))
        if len(sorted_pts) == 0:
            return sorted_pts
        roll_idx = int(np.argmax([
            center.x - p.x + center.y - p.y for p in sorted_pts
        ]))
        return np.roll(sorted_pts, -roll_idx).tolist()

    def merge(self, other: 'Polygon', force: bool = True) -> 'Polygon':
        if not force:
            intersection_area = self.get_intersection_area(other)
            if intersection_area == 0:
                raise ValueError("No intersection between polygons.")
            merged = self.geometric.union(other.geometric)
            points = list(set([Point(*el) for el in merged.exterior.coords]))
            return Polygon(points)
        merged = unary_union([self.geometric, other.geometric]).convex_hull
        points = list(set([Point(*el) for el in merged.exterior.coords]))
        return Polygon(points)

    def get_points_list(self) -> List:
        return [p.as_list() for p in self.points]

    def get_points_tuple(self) -> Tuple:
        return tuple(p.as_tuple() for p in self.points)

    def get_area(self) -> float:
        return self.geometric.area

    def get_intersection_area(self, other: 'Polygon') -> float:
        return self.geometric.intersection(other.geometric).area

    def get_union_area(self, other: 'Polygon') -> float:
        if not self.geometric.is_valid or not other.geometric.is_valid:
            return None
        return self.geometric.union(other.geometric).area

    def get_iou_score(self, other: 'Polygon') -> float:
        if self.geometric.intersects(other.geometric):
            return self.get_intersection_area(other) / self.get_union_area(other)
        return 0.0

    def get_inside_rate(self, other: 'Polygon') -> float:
        return self.get_intersection_area(other) / self.get_area()

    def to_dict(self) -> Dict:
        return {"points": [p.to_dict() for p in self.points]}

    @staticmethod
    def from_dict(d) -> 'Polygon':
        return Polygon([Point.from_dict(p) for p in d['points']])

    def __eq__(self, other):
        return self.points == other.points

    def __repr__(self):
        return f"Polygon({' '.join(repr(p) for p in self.points)})"


class BoundingBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = int(xmin)
        self.ymin = int(ymin)
        self.xmax = int(xmax)
        self.ymax = int(ymax)

    @property
    def center(self) -> Point:
        return Point(x=self.xcenter, y=self.ycenter)

    @property
    def xcenter(self) -> int:
        return (self.xmin + self.xmax) // 2

    @property
    def ycenter(self) -> int:
        return (self.ymin + self.ymax) // 2

    @property
    def width(self) -> int:
        return abs(self.xmax - self.xmin) + 1

    @property
    def height(self) -> int:
        return abs(self.ymax - self.ymin) + 1

    @property
    def corners(self) -> List[Point]:
        return [
            Point(self.xmin, self.ymin),
            Point(self.xmin, self.ymax),
            Point(self.xmax, self.ymax),
            Point(self.xmax, self.ymin),
        ]

    def get_pascal_voc_format(self) -> Tuple[int, int, int, int]:
        return self.xmin, self.ymin, self.xmax, self.ymax

    def get_area(self) -> int:
        return self.width * self.height

    def get_intersection_area(self, other: 'BoundingBox') -> int:
        iw = max(0, min(self.xmax, other.xmax) - max(self.xmin, other.xmin) + 1)
        ih = max(0, min(self.ymax, other.ymax) - max(self.ymin, other.ymin) + 1)
        return iw * ih

    def get_iou_score(self, other: 'BoundingBox') -> float:
        inter = self.get_intersection_area(other)
        return inter / (self.get_area() + other.get_area() - inter)

    def get_inside_rate(self, other: 'BoundingBox') -> float:
        return self.get_intersection_area(other) / self.get_area()

    def add_margin(self, margin_percent: float, max_bottom: int, max_right: int) -> 'BoundingBox':
        if not 0 <= margin_percent <= 1:
            margin_percent /= 100
        hpad = int(self.width * margin_percent)
        vpad = int(self.height * margin_percent)
        xmin = max(0, self.xmin - hpad)
        ymin = max(0, self.ymin - vpad)
        xmax = min(max_right, self.xmax + hpad)
        ymax = min(max_bottom, self.ymax + vpad)
        if xmax - xmin < 3 or ymax - ymin < 3:
            return BoundingBox(self.xmin, self.ymin, self.xmax, self.ymax)
        return BoundingBox(xmin, ymin, xmax, ymax)

    def rectify(self, T: np.ndarray) -> 'BoundingBox':
        edge_points = np.array([
            [(self.xmin + self.xmax) / 2, self.ymin],
            [self.xmax, (self.ymin + self.ymax) / 2],
            [(self.xmin + self.xmax) / 2, self.ymax],
            [self.xmin, (self.ymin + self.ymax) / 2],
        ], dtype=np.float32)
        pts_h = np.hstack([edge_points, np.ones((4, 1))])
        transformed = (T @ pts_h.T).T
        transformed = transformed[:, :2] / transformed[:, 2:3]
        xmin, ymin = int(transformed[:, 0].min()), int(transformed[:, 1].min())
        xmax, ymax = int(transformed[:, 0].max()), int(transformed[:, 1].max())
        if xmax - xmin < 3 or ymax - ymin < 3:
            return self
        self.xmin, self.ymin, self.xmax, self.ymax = xmin, ymin, xmax, ymax
        return self

    def to_dict(self) -> Dict:
        return {
            "xmin": round(float(self.xmin), 4),
            "ymin": round(float(self.ymin), 4),
            "xmax": round(float(self.xmax), 4),
            "ymax": round(float(self.ymax), 4),
        }

    @staticmethod
    def from_dict(d) -> 'BoundingBox':
        return BoundingBox(xmin=d['xmin'], ymin=d['ymin'], xmax=d['xmax'], ymax=d['ymax'])

    def __eq__(self, other):
        return (self.xmin == other.xmin and self.ymin == other.ymin and
                self.xmax == other.xmax and self.ymax == other.ymax)

    def __repr__(self):
        return f"BoundingBox({self.xmin}, {self.ymin}, {self.xmax}, {self.ymax})"


class Detection(BoundingBox):
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int,
                 score: float = 1.0, name: str = "product",
                 object_id: int = 0, embedding=None):
        super().__init__(xmin, ymin, xmax, ymax)
        self.score = score
        self.name = name
        self.object_id = object_id
        self.embedding = embedding

    def to_dict(self) -> Dict:
        d = super().to_dict()
        d.update({"score": float(self.score), "name": self.name, "object_id": self.object_id})
        return d

    @staticmethod
    def from_dict(d) -> 'Detection':
        return Detection(
            xmin=d['xmin'], ymin=d['ymin'], xmax=d['xmax'], ymax=d['ymax'],
            score=d.get('score', 1.0), name=d.get('name', 'product'),
            object_id=d.get('object_id', 0),
        )

    def __repr__(self):
        return f"{self.name}({self.score * 100:.1f}%)"


def collect_embeddings(detections: List[Detection]) -> List:
    return [d.embedding for d in detections if d.embedding is not None]
