import inspect
import logging

import numpy as np
import scipy.ndimage as spim

from porespy.tools import get_tqdm, parse_shape, settings

tqdm = get_tqdm()
logger = logging.getLogger(__name__)


__all__ = [
    'random_cantor_dust',
    'sierpinski_foam',
]


def random_cantor_dust(
    shape, n: int = 5,
    p: int = 2,
    f: float = 0.8,
    seed: int = None,
):
    r"""
    Generates an image of random cantor dust

    Parameters
    ----------
    shape : array_like
        The shape of the final image.  If not evenly divisible by $p**n$
        it will be increased to the nearest size that is.
    n : int
        The number of times to iteratively divide the image.
    p : int (default = 2)
        The number of divisions to make on each iteration.
    f : float (default = 0.8)
        The fraction of the set to keep on each iteration.
    seed : int, optional, default = `None`
        Initializes numpy's random number generator to the specified state. If not
        provided, the current global value is used. This means calls to
        ``np.random.state(seed)`` prior to calling this function will be respected.

    Returns
    -------
    dust : ndarray
        A boolean image of a random Cantor dust

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/random_cantor_dust.html>`__
    to view online example.

    """
    if seed is not None:
        np.random.seed(seed)
    # Parse the given shape and adjust if necessary
    shape = parse_shape(shape)
    trim = np.mod(shape, (p**n))
    if np.any(trim > 0):
        shape = shape - trim + p**n
        logger.warning(f"Requested shape being changed to {shape}")
    im = np.ones(shape, dtype=bool)
    divs = []
    if isinstance(n, int):
        for i in range(1, n):
            divs.append(p**i)
    else:
        for i in n:
            divs.append(p**i)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i in tqdm(divs, desc=desc, **settings.tqdm):
        sh = (np.array(im.shape)/i).astype(int)
        mask = np.random.rand(*sh) < f
        mask = spim.zoom(mask, zoom=i, order=0)
        im = im*mask
    return im


def sierpinski_foam(
    shape,
    n: int = 5,
    mode: str = 'upper',
):
    r"""
    Generates an image of a Sierpinski carpet or foam with independent control of
    image size and number of iterations

    Parameters
    ----------
    shape : array_like
        The shape of the final image to create. To create a full image with no
        cropping, use a that is a multiple of `3**n`.
    n : int
        The number of times to iteratively divide the image. This functions starts
        by inserting single voxels, then inserts increasingly large squares/cubes.
    mode : str
        Controls the portion of the image that is returned, options are:

        ============= ==============================================================
        Mode          Description
        ============= ==============================================================
        `'upper'`     Returns the upper corner
        `'centered'`  Returns the center portion
        `None`        Provide the full image, in which case the returned image will
                      be larger than `shape`.
        ============= ==============================================================

    Returns
    -------
    im : ndarray
        A boolean image with `False` values inserted at the center of each
        square (or cubic) sub-section.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/generators/reference/sierpinski_foam.html>`__
    to view online example.

    """
    shape = parse_shape(shape)
    m = n
    if 3**(n+1)//3 < max(shape):
        while 3**(m+1)//3 < max(shape):
            m += 1
    im = np.zeros([3**(m+1)//3 for _ in range(len(shape))], dtype=bool)
    i = 0
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    pbar = tqdm(desc=desc, **settings.tqdm)
    while i < n:
        if im.ndim == 2:
            mask = np.zeros([3**(i+1), 3**(i+1)], dtype=bool)
            s = 3**(i+1)//3
            mask[s:-s, s:-s] = 1
            t = int(np.ceil(im.shape[0]/mask.shape[0]))
            im2 = np.tile(mask, [t, t])
        if im.ndim == 3:
            mask = np.zeros([3**(i+1), 3**(i+1), 3**(i+1)], dtype=bool)
            s = 3**(i+1)//3
            mask[s:-s, s:-s, s:-s] = 1
            t = int(np.ceil(im.shape[0]/mask.shape[0]))
            im2 = np.tile(mask, [t, t, t])
        im += im2
        i += 1
        pbar.update()
    pbar.close()

    if mode is None:
        slices = [...]
    elif mode == 'centered':
        slices = [slice(im.shape[ax]//2 - shape[ax]//2,
                        im.shape[ax]//2 + shape[ax]//2,
                        None) for ax in range(im.ndim)]
    elif mode == 'upper':
        slices = [slice(0, shape[ax], None) for ax in range(im.ndim)]
    im = im[tuple(slices)]
    im = im == 0  # Invert image
    return im
