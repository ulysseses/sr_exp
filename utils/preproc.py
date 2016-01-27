from __future__ import division, absolute_import, print_function
from six.moves import range, zip

import os
import time
import numpy as np
import scipy.misc as sm
from scipy.ndimage.filters import gaussian_filter

import h5py
from fuel.datasets.hdf5 import H5PYDataset


def unit2byte(img_unit):
    """Convert `img_unit` from floating [0, 1] to integer [0, 255]"""
    return np.clip(np.round(255*img_unit), 0, 255).astype('uint8')


def byte2unit(img_byte):
    """Convert `img_byte`from integer [0, 255] to floating [0, 1]"""
    return img_byte.astype('float32') / 255.


def imresize(img, s):
    """
    Resize routine. If `s` < 1, then filter the image with a gaussian
    to prevent aliasing artifacts.

    Args:
      img: uint8 image
      s: down/up-sampling factor

    Returns:
      img: uint8 resized & bi-cubic interpolated image
    """
    #if s < 1:
    #    img = gaussian_filter(img, 0.5)
    img = sm.imresize(img, s, interp='bicubic')
    return img


def shave(img, border):
    """Shave off border from image.

    Args:
      img: image
      border: border

    Returns:
      img: img cropped of border
    """
    return img[border: -border, border: -border]


def padZeroBorder(img, border=3):
    """Pad image with a border of zeros

    Args:
      img: image
      border (3): border

    Returns:
      img_new: zero-padded `img`
    """
    h, w = img.shape
    img_new = np.zeros((h + 2*border, w + 2*border), dtype=img.dtype)
    img_new[border:-border, border:-border] = img
    return img_new


def rgb2ycc(img_rgb):
    """Convert image from rgb to ycbcr colorspace.

    Args:
      img_rgb: image in rgb colorspace

    Returns:
      img_ycc: image in ycbcr colorspace
    """
    img_rgb = img_rgb.astype('float32')
    img_ycc = np.empty_like(img_rgb, dtype='float32')
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    y, cb, cr = img_ycc[:,:,0], img_ycc[:,:,1], img_ycc[:,:,2]

    y[:] = .299*r + .587*g + .114*b
    cb[:] = 128 -.168736*r -.331364*g + .5*b
    cr[:] = 128 +.5*r - .418688*g - .081312*b

    img_ycc = np.clip(np.round(img_ycc), 0, 255).astype('uint8')
    return img_ycc


def ycc2rgb(img_ycc):
    """Convert image from ycbcr to rgb colorspace.

    Args:
      img_ycc: uint8 image in ycbcr colorspace

    Returns:
      img_rgb: uint8 image in rgb colorspace
    """
    img_ycc = img_ycc.astype('float32')
    img_rgb = np.empty_like(img_ycc, dtype='float32')
    y, cb, cr = img_ycc[:,:,0], img_ycc[:,:,1], img_ycc[:,:,2]
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]

    r[:] = y + 1.402 * (cr-128)
    g[:] = y - .34414 * (cb-128) -  .71414 * (cr-128)
    b[:] = y + 1.772 * (cb-128)

    img_rgb = np.clip(np.round(img_rgb), 0, 255).astype('uint8')
    return img_rgb


def modcrop(img, modulo):
    """
    Crop `img` s.t. its dimensions are an integer multiple of `modulo`

    Args:
      img: image
      modulo: modulo factor

    For example:

    ```
    # 'img' is [[1, 2, 3], [4, 5, 6],
    #           [7, 8, 9], [1, 2, 3],
    #           [4, 5, 6], [7, 8, 9]]
    modcrop(img, 2) ==> [[1, 2, 3], [4, 5, 6],
                         [7, 8, 9], [1, 2, 3]]
    ```
    """
    h, w = img.shape[0], img.shape[1]
    h = (h // modulo) * modulo
    w = (w // modulo) * modulo
    img = img[:h, :w]
    return img


def padcrop(img, modulo):
    """
    Pad `img` s.t. its dimensions are an integer multiple of `modulo`

    Args:
      img: image
      modulo: modulo factor

    For example:

    ```
    # 'img' is [[1, 2, 3], [4, 5, 6],
    #           [7, 8, 9], [1, 2, 3],
    #           [4, 5, 6], [7, 8, 9]]
    modcrop(img, 2) ==> [[1, 2, 3], [4, 5, 6],
                         [7, 8, 9], [1, 2, 3],
                         [4, 5, 6], [7, 8, 9],
                         [0, 0, 0], [0, 0, 0]]
    ```
    """
    h, w = img.shape[0], img.shape[1]
    h2, w2 = np.ceil(h / modulo) * modulo, np.ceil(w / modulo) * modulo
    shp2 = list(img.shape)
    shp2[0] = h2
    shp2[1] = w2
    img2 = np.zeros(shp2, dtype=img.dtype)
    img2[:h, :w] = img
    return img2


def _crop_gen(img, cw, s):
    """
    Generate a strided series of cropped patches from an image.

    Args:
      img: image
      cw: crop width
      s: stride

    Yields:
      crop: 2d cropped region of `img`
    """
    for i in range(0, img.shape[0] - cw + 1, s):
        for j in range(0, img.shape[1] - cw + 1, s):
            crop = img[i : i + cw, j : j + cw]
            yield crop


def _num_crops(img, cw, s, tup=False):
    h, w = img.shape[0], img.shape[1]
    n_y, n_x = len(range(0, h - cw + 1, s)), len(range(0, w - cw + 1, s))
    if tup:
        return n_y, n_x
    else:
        return n_y * n_x


def _get_filenames(path):
    ext_set = set(['jpg', 'jpeg', 'png', 'bmp'])
    def _valid_ext(s):
        lst = s.split('.')
        if len(lst) != 2: return False
        sfx = lst[1].lower()
        return sfx in ext_set

    fns = [os.path.join(path, fn) for fn in os.listdir(path)
           if _valid_ext(fn)]
    return fns


def _prepare_hdf5(path_h5, n_tr, n_te, n_va, cw, **kwargs):
    f = h5py.File(path_h5, mode='w')
    shp = (n_tr + n_te + n_va, cw, cw, 1)
    split_dict = {
        'train': {'LR': (0, n_tr), 'HR': (0, n_tr)},
        'test':  {'LR': (n_tr, n_tr + n_te),
                  'HR': (n_tr, n_tr + n_te)},
        'val': {'LR': (n_tr + n_te, n_tr + n_te + n_va),
                'HR': (n_tr + n_te, n_tr + n_te + n_va)}
    }
    f.attrs['split'] = H5PYDataset.create_split_array(split_dict)

    LRh5 = f.create_dataset('LR', shp, dtype=np.float32, **kwargs)
    HRh5 = f.create_dataset('HR', shp, dtype=np.float32, **kwargs)

    return f, LRh5, HRh5


def _chunk_gen(start, stop, chunk_size):
    for i in range(start, stop, chunk_size):
        j = min(i + chunk_size, stop)
        yield i, j


def _store_crops(lrs, hrs, img_lr, img_hr, crop_ind, f, LRh5, HRh5,
                 chunk_size, cw, stride, prune, cutoff=None):
    crop_ind0 = crop_ind
    ind = 0
    for crop_lr, crop_hr in zip(_crop_gen(img_lr, cw, stride),
                                _crop_gen(img_hr, cw, stride)):
        if (cutoff and np.var(crop_hr) > cutoff) or prune == 0:
            lrs[ind] = crop_lr[..., np.newaxis]
            hrs[ind] = crop_hr[..., np.newaxis]
            
            ind += 1
            crop_ind += 1
        if crop_ind % chunk_size == 0:
            LRh5[crop_ind0 : crop_ind] = lrs[:ind]
            HRh5[crop_ind0 : crop_ind] = hrs[:ind]
            f.flush()
            crop_ind0 = crop_ind
            ind = 0
    LRh5[crop_ind0 : crop_ind] = lrs[:ind]
    HRh5[crop_ind0 : crop_ind] = hrs[:ind]
    f.flush()
    return crop_ind


def _find_cutoff(fns, n, prune, sr, border, cw, stride):
    # Collect variances
    X_var = np.empty(n, dtype='float32')
    ind = 0
    for fn in fns:
        img = rgb2ycc(sm.imread(fn))[:, :, 0]  # rgb --> y
        img = byte2unit(img)
        img_hr = shave(modcrop(img, sr), border)
        for crop in _crop_gen(img_hr, cw, stride):
            X_var[ind] = np.var(crop)
            ind += 1

    # Find var cutoff point
    bottom = int(prune * n)
    cutoff = X_var[np.argpartition(X_var, bottom)[:bottom]].max()

    # Return cutoff & new_n
    mask = X_var > cutoff
    new_n = np.sum(mask)
    return cutoff, new_n


def store_hdf5(conf, pp=None, **kwargs):
    """
    Generate crops from images in a directory and store into hdf5. Optionally
    preprocess them with and/or prune low-variance crops.

    Args:
      conf:
        path_h5: path to write hdf5 file
        path_tr: path to training images
        path_te: path to testing images
        path_va: path to validation images
        cw: crop width
        stride: stride
        sr: resize/super-resolution factor
        border: how much to shave off before resizing down `img_hr` --> `img_lr`
        augment: if True, augment dataset with rotations and flips
        prune: remove the `prune`%% lowest variance _num_crop
        chunk_size: number of rows to save in memory before flushing hdf5
      pp (None): if not None, function/lambda to preprocess image
      **kwargs: passed to h5py.File
    """
    path_h5             = conf['path_h5']
    path_tr             = conf['path_tr']
    path_te             = conf['path_te']
    path_va             = conf['path_va']
    cw                  = conf['cw']
    stride              = conf['stride']
    sr                  = conf['sr']
    augment             = conf['augment']
    prune               = conf['prune']
    border              = conf['border']
    chunk_size          = conf['chunk_size']
    data_cached         = conf['data_cached']

    if data_cached:
        return

    # Count number of training/testing examples
    fns_tr = _get_filenames(path_tr)
    fns_te = _get_filenames(path_te)
    fns_va = _get_filenames(path_va)
    
    #fns_tr = fns_tr[:len(fns_tr) // 16]
    #fns_te = fns_te[:len(fns_te) // 32]
    #fns_va = fns_va[:len(fns_va) // 32]

    n_tr, n_te, n_va = 0, 0, 0
    for fn in fns_tr:
        img = shave(modcrop(sm.imread(fn), sr), border)
        n_tr += _num_crops(img, cw, stride)
    for fn in fns_te:
        img = shave(modcrop(sm.imread(fn), sr), border)
        n_te += _num_crops(img, cw, stride)
    for fn in fns_va:
        img = shave(modcrop(sm.imread(fn), sr), border)
        n_va += _num_crops(img, cw, stride)

    # Prune out low-variance crops
    if prune > 0:
        cutoff, n = _find_cutoff(fns_tr, n_tr, prune, sr, border, cw, stride)
        n_actual = n
    else:
        cutoff, n = None, n_tr
        n_actual = n

    # Prepare fuel/h5py file
    n_actual = 4*n_actual if augment else n_actual
    f, LRh5, HRh5 = _prepare_hdf5(path_h5, n_actual, n_te, n_va, cw,
                                  **kwargs)
    
    print("n_tr:", n_actual)
    print("n_te:", n_te)
    print("n_va:", n_va)
    #print("n:", n_actual + n_te + n_va)
    
    crop_ind = 0
    lrs = np.empty((chunk_size, cw, cw, 1), dtype=np.float32)
    hrs = np.empty((chunk_size, cw, cw, 1), dtype=np.float32)
    for fn in fns_tr:
        img = rgb2ycc(sm.imread(fn))[:, :, 0]  # rgb --> y
        img_lr, img_hr = lr_hr(img, sr, border)
        img_lr, img_hr = byte2unit(img_lr), byte2unit(img_hr)
        crop_ind = _store_crops(lrs, hrs, img_lr, img_hr, crop_ind, f, LRh5, HRh5,
                                chunk_size, cw, stride, prune, cutoff)
    f.flush()
    assert crop_ind == n
    
    # Augment with rotations and flips
    if augment:
        for fn in fns_tr:
            img = rgb2ycc(sm.imread(fn))[:, :, 0]  # rgb --> y
            img_lr, img_hr = lr_hr(img, sr, border)
            img_lr, img_hr = byte2unit(img_lr), byte2unit(img_hr)

            aug_lr, aug_hr = np.rot90(img_lr), np.rot90(img_hr)
            crop_ind = _store_crops(lrs, hrs, aug_lr, aug_hr, crop_ind, f, LRh5, HRh5,
                                    chunk_size, cw, stride, prune, cutoff)
            
            aug_lr, aug_hr = np.flipud(img_lr), np.flipud(img_hr)
            crop_ind = _store_crops(lrs, hrs, aug_lr, aug_hr, crop_ind, f, LRh5, HRh5,
                                    chunk_size, cw, stride, prune, cutoff)
                       
            aug_lr, aug_hr = np.rot90(np.flipud(img_lr)), np.rot90(np.flipud(img_hr))
            crop_ind = _store_crops(lrs, hrs, aug_lr, aug_hr, crop_ind, f, LRh5, HRh5,
                                    chunk_size, cw, stride, prune, cutoff)
        f.flush()
        assert crop_ind >= n_actual
        crop_ind = n_actual
    
    for fn in fns_te + fns_va:
        img = rgb2ycc(sm.imread(fn))[:, :, 0]  # rgb --> y
        img_lr, img_hr = lr_hr(img, sr, border)
        img_lr, img_hr = byte2unit(img_lr), byte2unit(img_hr)
        crop_ind = _store_crops(lrs, hrs, img_lr, img_hr, crop_ind, f, LRh5, HRh5,
                                chunk_size, cw, stride, 0)
    f.close()
    assert crop_ind == n_actual + n_te + n_va


def lr_hr(img, sr, border=3):
    """Generate LR & HR pair from image.

    Args:
      img: uint8 image
      sr: resize/super-resolution factor
      border (3): how much to shave off before resizing down `img_hr` --> `img_lr`
    """
    img_hr = modcrop(img, sr)
    down_shp = (img_hr.shape[0] // sr, img_hr.shape[1] // sr)
    img_lr = imresize(img_hr, down_shp)
    img_lr = imresize(img_lr, img_hr.shape)
    img_lr = shave(img_lr, border)
    img_hr = shave(img_hr, border)
    return img_lr, img_hr
