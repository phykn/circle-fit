import cv2
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from sklearn.neighbors import LocalOutlierFactor
from typing import Union, Optional, Tuple


def imread(
    path: str
) -> np.ndarray:
    
    return Image.open(path).convert('RGB')


def canny(
    img: Union[Image.Image, np.ndarray],
    thresh: int = 127
) -> np.ndarray:
    
    if isinstance(img, Image.Image):
        img = np.array(img)

    # to gray
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
        threshold1 = 100,
        threshold2 = 200
    )

    return img


def convex(
    img: np.ndarray,
    n_neighbors: int = 3,
    n_sample: Optional[int] = None,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    
    # gather edge data
    y, x = np.where(img == 255)
    xy = np.stack([x, y], axis = 1)

    # sampling if need
    if n_sample is not None:
        if len(xy) > n_sample:
            np.random.seed(random_state)
            index = np.arange(len(xy))
            index = np.random.choice(index, size = n_sample)
            xy = xy[index]

    # convex
    hull = ConvexHull(points = xy)
    xy = xy[np.unique(hull.simplices)]

    # drop outlier
    p = LocalOutlierFactor(
        n_neighbors = np.minimum(n_neighbors, len(xy))
    ).fit_predict(xy)

    xy = xy[np.where(p == 1)]

    # get xy
    x, y = xy.T
    return x, y