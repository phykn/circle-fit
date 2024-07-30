from PIL import Image
from typing import Tuple, Optional
from .image import imread, canny, convex
from .taubin import taubin_method


def fit_circle(
    img: Image.Image,
    thresh: int = 50,
    n_neighbors: int = 10,
    n_sample: Optional[int] = None,
    random_state: int = 42
) -> Tuple[float, float, float]:
    
    """
    Detect and fit a circle in an image.

    This function processes an input image to detect edges, find points that 
    approximate a circle, and then fits a circle to these points using the 
    Taubin method. The following steps are performed:

    1. Apply Canny edge detection to the input image using the specified 
       threshold value.
    2. Use the convex hull approach to identify points that are likely 
       to form a circle. This is controlled by the number of neighbors 
       and samples, which can be adjusted for precision and performance.
    3. Fit a circle to the identified points using the Taubin method, 
       returning the circle's center coordinates (x0, y0) and radius (r).

    Parameters:
    img (Image.Image): The input image in which to detect and fit the circle.
    thresh (int): Threshold value for Canny edge detection. Default is 50.
    n_neighbors (int): Number of neighbors to consider when finding points 
                       for convex hull. Default is 10.
    n_sample (Optional[int]): Number of points to sample for fitting. If None, 
                              all points are used. Default is None.
    random_state (int): Seed for random number generator for reproducibility. 
                        Default is 42.

    Returns:
    Tuple[float, float, float]: The x and y coordinates of the circle's center 
                                and the radius of the fitted circle.
    """

    img = canny(
        img, 
        thresh = thresh
    )

    x, y = convex(
        img = img, 
        n_neighbors = n_neighbors,
        n_sample = n_sample,
        random_state = random_state
    )

    x0, y0, r = taubin_method(x, y)
    return x0, y0, r