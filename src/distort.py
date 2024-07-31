import skimage
import numpy as np
from scipy import ndimage


def resize(
    img: np.ndarray, 
    scale: int = 1,
    order: int = 1,
    mode: str = 'reflect',
    anti_aliasing: bool = False
) -> np.ndarray:
    
    img = img.astype(float)

    vmax = np.max(img)
    img = img / vmax

    shape = scale * np.array(img.shape)

    img = skimage.transform.resize(
        image = img,
        output_shape = shape,
        order = order,
        mode = mode,
        anti_aliasing = anti_aliasing        
    )

    img = vmax * img
    return img


def distort(
    img: np.ndarray, 
    h0: float, 
    w0: float, 
    r: float, 
    scale: int = 1,
    order: int = 1,
    mode: str = 'reflect',
    anti_aliasing: bool = False
) -> np.ndarray:
    
    # store init r
    r0 = int(np.floor(r))

    # rescale r
    r = r / scale

    # crop image
    h, w = img.shape

    hmin = np.maximum(0, round(h0 - r))
    hmax = np.minimum(h, round(h0 + r))
    wmin = np.maximum(0, round(w0 - r))
    wmax = np.minimum(w, round(w0 + r))

    _h, _w = np.indices(img.shape)

    _i = img[hmin:hmax, wmin:wmax]
    _h = _h[hmin:hmax, wmin:wmax]
    _w = _w[hmin:hmax, wmin:wmax]

    # resize
    _i = resize(
        img = _i, 
        scale = scale,
        order = order,
        mode = mode,
        anti_aliasing = anti_aliasing
    )

    _h = resize(
        img = _h, 
        scale = scale,
        order = order,
        mode = mode,
        anti_aliasing = anti_aliasing
    )

    _w = resize(
        img = _w, 
        scale = scale,
        order = order,
        mode = mode,
        anti_aliasing = anti_aliasing
    )

    # r, t coordinate
    rr = np.sqrt((_h - h0) ** 2 + (_w - w0) ** 2)
    rr = np.round(rr).astype(int)

    tt = np.arctan2(_h - h0, _w - w0)
    tt = (tt + np.pi) * 180 / np.pi
    tt = np.round(tt).astype(int)
    tt = np.mod(tt, 360)

    # distort
    hh, ww = np.indices(_i.shape)

    new_img = np.full(
        shape = (np.max(rr) + 1, 360), 
        fill_value = np.nan
    )

    new_img[rr.tolist(), tt.tolist()] = _i[hh.tolist(), ww.tolist()]

    # padding
    _r = new_img.shape[0]

    if _r < r0:
        new_img = np.pad(
            array = new_img, 
            pad_width = ((0, r0 - _r), (0, 0)),
            constant_values = np.nan
        )
    else:
        new_img = new_img[:r0, :]

    new_img = np.round(new_img)
    new_img = np.clip(new_img, 0, 255)
    return new_img


def nan_to_num(
    img: np.ndarray, 
    filter_size: int = 9,
) -> np.ndarray:
    
    filled_img = np.copy(img)
    filled_img[np.isnan(filled_img)] = 0
    filled_img = ndimage.median_filter(filled_img, size = filter_size)
    img[np.isnan(img)] = filled_img[np.isnan(img)]
    return img


def fast_distort(
    img: np.ndarray, 
    h0: float, 
    w0: float, 
    r: float, 
    num_iter: int = 1, 
    order: int = 1,
    mode: str = 'reflect',
    anti_aliasing: bool = False,    
    filter_size: int = 9
) -> np.ndarray:
    
    new_img = None

    # distort
    for i in range(num_iter):
        if new_img is None:
            _img = distort(
                img = img, 
                h0 = h0, 
                w0 = w0, 
                r = r, 
                scale = 2 ** i,
                order = order,
                mode = mode,
                anti_aliasing = anti_aliasing
            )

            new_img = _img
        else:
            _img = distort(
                img = img, 
                h0 = h0, 
                w0 = w0, 
                r = r, 
                scale = 2 ** i,
                order = order,
                mode = mode,
                anti_aliasing = anti_aliasing
            )
            
            new_img = np.where(np.isnan(new_img), _img, new_img)

        num_nan = np.sum(np.isnan(new_img))
        if num_nan == 0:
            break

    # fill nan
    if num_nan > 0:
        new_img = nan_to_num(
            img = new_img, 
            filter_size = filter_size
        )
                
    return new_img