import numpy as np
from typing import Tuple


def taubin_method(
    x: np.ndarray, 
    y: np.ndarray
) -> Tuple[float, float, float]:
    
    assert x.ndim == 1, "x must be a 1-dimensional array."
    assert y.ndim == 1, "y must be a 1-dimensional array."

    z = np.ones(x.shape, dtype = float)
    A = np.stack([x, y, z], axis = 1)
    b = x ** 2 + y ** 2
    out, _, _, _ = np.linalg.lstsq(A, b, rcond = None)

    x0 = out[0] / 2
    y0 = out[1] / 2
    r = np.sqrt(out[2] + x0 ** 2 + y0 ** 2)
    return x0, y0, r