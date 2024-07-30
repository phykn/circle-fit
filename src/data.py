import numpy as np
from typing import Tuple


def draw_circle(
    x0: float, 
    y0: float, 
    r: float, 
    num: int = 1000
) -> Tuple[np.ndarray, np.ndarray]:
    
    theta = np.linspace(0, 2 * np.pi, num = num)
    x = r * np.cos(theta) + x0
    y = r * np.sin(theta) + y0
    return x, y