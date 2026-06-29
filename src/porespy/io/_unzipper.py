import inspect
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import imageio
import numpy as np
from skimage.io import imread_collection

from porespy.tools import get_tqdm, settings

tqdm = get_tqdm()


__all__ = [
    "zip_to_stack",
    "folder_to_stack",
]


def folder_to_stack(target_dir):
    r"""
    Opens all images found in the target directory as single 3D numpy array

    Parameters
    ----------
    target_dir : str or path object
        The location of the folder containing the images.

    Returns
    -------
    im : ndarray
        A 3D numpy array of the imported image.

    Notes
    -----
    The files should be named with leading numerical values indicating their
    layer number, like 001, 002, etc.
    """
    p = Path(target_dir)
    test_im = imageio.v2.imread(os.path.join(p, os.listdir(p)[0]))
    im = np.zeros(
        shape=[
            test_im.shape[0],
            test_im.shape[1],
            len(os.listdir(p)),
        ],
        dtype=test_im.dtype,
    )
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, f in enumerate(tqdm(os.listdir(p), desc=desc, **settings.tqdm)):
        im[..., i] = imageio.v2.imread(os.path.join(p, f))

    return im


def zip_to_stack(f):
    r"""
    Reads a zip file containing 2D slices of a 3D image, and converts to a 3D stack

    Parameters
    ----------
    f : str
        The path and/or file name of the zip archive.  If ``<name>.zip`` is given,
        then it's assumed to be located in the current working directory. Otherwise
        a full path should be given, like ``C:\path\to\file.zip``. Either way, the
        archive is extracted into a folder in the given directory.

    Returns
    -------
    im : ndarray
        A 3D numpy array of the imported image.

    Notes
    -----
    The files should be named with leading numerical values indicating their
    layer number, like 001, 002, etc.
    """
    p = Path(f)
    target_dir = p.parts[-1].rpartition(".")[0]

    with ZipFile(p, "r") as f:
        f.extractall(target_dir)

    filenames = []
    for f in os.listdir(target_dir):
        filenames.append(os.path.join(target_dir, f))
    files = imread_collection(filenames, conserve_memory=True)
    im = np.stack(files, axis=2)

    # Remove the unzipped folder
    shutil.rmtree(target_dir)

    return im
