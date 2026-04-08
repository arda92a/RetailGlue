import cv2
import numpy as np
from PIL import Image


def read_image(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_pil(image: np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    return Image.fromarray(image)
