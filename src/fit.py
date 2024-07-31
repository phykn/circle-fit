import cv2
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from sklearn.neighbors import LocalOutlierFactor
from typing import Union, Optional, Tuple


def canny(
    img: Union[Image.Image, np.ndarray],
    thresh: int = 127
) -> np.ndarray:
    
    if isinstance(img, Image.Image):
        img = np.array(img)

    # to gray
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # bg bias
    img = img.astype(float)
    bg = np.mean([img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]])
    img = np.round(abs(img - bg))

    # drop outlier
    vmin = np.percentile(img, q = 1)
    vmax = np.percentile(img, q = 99)
    img = np.clip(img, vmin, vmax)
    img = img.astype(np.uint8)

    # normalize
    img = cv2.normalize(
        src = img, 
        dst = None, 
        alpha = 0, 
        beta = 255, 
        norm_type = cv2.NORM_MINMAX
    )

    # binary
    _, img = cv2.threshold(
        src = img, 
        thresh = thresh, 
        maxval = 255, 
        type = cv2.THRESH_BINARY
    )

    # canny
    img = cv2.Canny(
        image = img,
        threshold1 = 127,
        threshold2 = 255
    )

    return img


def convex(
    h: np.ndarray,
    w: np.ndarray,
    n_neighbors: int = 3,
    n_sample: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    
    hw = np.stack([h, w], axis = 1)

    # sampling if need
    if n_sample is not None:
        if len(hw) > n_sample:
            np.random.seed(random_state)
            index = np.arange(len(hw))
            index = np.random.choice(index, size = n_sample)
            hw = hw[index]

    # convex
    hull = ConvexHull(points = hw)
    hw = hw[np.unique(hull.simplices)]

    # drop outlier
    p = LocalOutlierFactor(
        n_neighbors = np.minimum(n_neighbors, len(hw))
    ).fit_predict(hw)

    hw = hw[np.where(p == 1)]

    # get hw
    h, w = hw.T
    return h, w


def taubin_method(
    h: np.ndarray, 
    w: np.ndarray
) -> Tuple[float, float, float]:
    
    assert h.ndim == 1, "x must be a 1-dimensional array."
    assert w.ndim == 1, "y must be a 1-dimensional array."

    z = np.ones(h.shape, dtype = float)
    A = np.stack([h, w, z], axis = 1)
    b = h ** 2 + w ** 2
    out, _, _, _ = np.linalg.lstsq(A, b, rcond = None)

    h0 = out[0] / 2
    w0 = out[1] / 2
    r = np.sqrt(out[2] + h0 ** 2 + w0 ** 2)
    return h0, w0, r


def fit_circle(
    img: Image.Image,
    thresh: int = 50,
    n_neighbors: int = 10,
    n_sample: Optional[int] = None,
    random_state: int = 42
) -> Tuple[float, float, float]:

    img = canny(
        img, 
        thresh = thresh
    )

    h, w = np.where(img == 255)

    h, w = convex(
        h = h,
        w = w,
        n_neighbors = n_neighbors,
        n_sample = n_sample,
        random_state = random_state
    )

    h0, w0, r = taubin_method(h, w)
    return h0, w0, r