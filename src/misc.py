import numpy as np
from PIL import Image
from typing import Tuple


def imread(
    path: str
) -> np.ndarray:
    
    return Image.open(path).convert('RGB')


def draw_circle(
    h0: float, 
    w0: float, 
    r: float, 
    num: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    
    theta = np.linspace(0, 2 * np.pi, num = num)
    h = r * np.sin(theta) + h0
    w = r * np.cos(theta) + w0    
    return w, h