import inspect
import logging
import math
import os
from pathlib import Path

import dask
import dask.array as da
import dask.distributed
import numba as nb
import numpy as np
import scipy.ndimage as spim
import scipy.spatial as sptl
from skimage.morphology import footprint_rectangle
from skimage.segmentation import watershed

from porespy.tools import (
    Results,
    _check_for_singleton_axes,
    extend_slice,
    get_edt,
    get_tqdm,
    ps_rect,
    ps_round,
    settings,
)

from ._funcs import chunked_func

__all__ = [
    "snow_partitioning",
    "snow_partitioning_n",
    "snow_partitioning_parallel",
    "find_peaks",
    "reduce_peaks",
    "trim_nearby_peaks",
    "trim_saddle_points",
    "trim_saddle_points_legacy",
]


edt = get_edt()
tqdm = get_tqdm()
logger = logging.getLogger(__name__)


def snow_partitioning(im, dt=None, r_max=4, sigma=0.4, peaks=None, mask_watershed=False):
    r"""
    Partition the void space into pore regions using a marker-based
    watershed algorithm

    Parameters
    ----------
    im : array_like
        A boolean image of the domain, with ``True`` indicating the pore
        space and ``False`` elsewhere.
    dt : array_like, optional
        The distance transform of the pore space.  This is done
        automatically if not provided, but if the distance transform has
        already been computed then supplying it can save some time.
    r_max : int
        The radius of the spherical structuring element to use in the
        Maximum filter stage that is used to find peaks. The default is 4.
    sigma : float
        The standard deviation of the Gaussian filter used in step 1. The
        default is 0.4.  If 0 is given then the filter is not applied,
        which is useful if a distance transform is supplied as the ``im``
        argument that has already been processed.
    peaks : ndarray, optional
        Optionally, it is possible to supply an array containing peaks, which
        are used as markers in the watershed segmentation. If a boolean array
        is received (``True`` indicating peaks), then ``scipy.ndimage.label``
        with cubic connectivity is used to label them. If an integer array is
        received then it is assumed the peaks have already been labelled.
        This allows for comparison of peak finding algorithms for instance.
        If this argument is provided, then ``r_max`` and ``sigma`` are ignored
        since these are specfically used in the peak finding process.
    mask_watershed : bool
        If True the watershed floods only the phase of interest. This is
        several times faster (the solid is not flooded) and pore components
        without any marker are left as 0 instead of receiving the label of
        an unrelated pore through the solid. Default False (legacy result).

    Returns
    -------
    results : Results object
        A custom object with the following data as attributes:

        ============ ==========================================================
        Item         Description
        ============ ==========================================================
        ``im``       The binary image of the void space
        ``dt``       The distance transform of the image
        ``peaks``    The peaks of the distance transform after applying the
                     steps of the SNOW algorithm
        ``regions``  The void space partitioned into pores using a marker
                     based watershed with the peaks found by the SNOW algorithm
        ============ ==========================================================

    Notes
    -----
    The SNOW network extraction algorithm (Sub-Network of an
    Over-segmented Watershed) was designed to handle to perculiarities of
    high porosity materials, but it applies well to other materials as
    well.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction
    algorithm using marker-based watershed segmenation".  Physical Review
    E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/snow_partitioning.html>`__
    to view online example.

    """
    logger.info("Beginning SNOW algorithm")
    im_shape = np.array(im.shape)
    if im.dtype is not bool:
        logger.info("Converting supplied image to boolean")
        im = im > 0
    if dt is None:
        logger.info("Performing distance transform")
        if np.any(im_shape == 1):
            dt = edt(im.squeeze())
            dt = dt.reshape(im_shape)
        else:
            dt = edt(im)

    if peaks is None:
        if sigma > 0:
            logger.info(f"Applying Gaussian blur with sigma = {sigma}")
            dt_blurred = spim.gaussian_filter(input=dt, sigma=sigma)*im
        else:
            dt_blurred = np.copy(dt)
        peaks = find_peaks(dt=dt_blurred, r_max=r_max)

        logger.debug(f"Initial number of peaks: {spim.label(peaks)[1]}")
        peaks = trim_saddle_points(peaks=peaks, dt=dt)
        logger.debug(f"Peaks after trimming saddle points: {spim.label(peaks)[1]}")
        if len(peaks) > 0:
            peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
            logger.debug(f"Peaks after trimming nearby points: {spim.label(peaks)[1]}")
            peaks, N = spim.label(peaks > 0, structure=ps_rect(3, dt.ndim))
            regions = watershed(image=-dt, markers=peaks,
                                mask=im if mask_watershed else None)
        else:
            regions = np.ones_like(dt_blurred)
    else:
        peaks, N = spim.label(peaks > 0, structure=ps_rect(3, im.ndim))
        regions = watershed(image=-dt, markers=peaks,
                            mask=im if mask_watershed else None)
    tup = Results()
    tup.im = im
    tup.dt = dt
    tup.peaks = peaks
    tup.regions = regions * (im > 0)
    return tup


def snow_partitioning_n(im, r_max=4, sigma=0.4, peaks=None):
    r"""
    This function partitions an imaging oontain an arbitrary number of
    phases into regions using a marker-based watershed segmentation.

    Parameters
    ----------
    im : ndarray
        Image of porous material where each phase is represented by unique
        integer starting from 1 (0's are ignored).
    r_max : scalar
        The radius of the spherical structuring element to use in the
        Maximum filter stage that is used to find peaks. The default is 4.
    sigma : scalar
        The standard deviation of the Gaussian filter used. The default is
        0.4. If 0 is given then the filter is not applied.
    peaks : ndarray, optional
        Optionally, it is possible to supply an array containing peaks, which
        are used as markers in the watershed segmentation. Must be a boolean
        array with ``True`` indicating peaks; ``scipy.ndimage.label``
        with cubic connectivity is used to label them. If this argument is
        provided then ``r_max`` and ``sigma`` are ignored, since these are
        specfically used in the peak finding process.

    Returns
    -------
    results : Results object
        A custom object with the following data as attributes:

        - 'im'
            The original image of the porous material

        - 'dt'
            The combined distance transform in alll phases of the image

        - 'phase_max_label'
            The list of max label of each phase in order to
            distinguish between each other

        - 'regions'
            The partitioned regions of n phases using a marker
            based watershed with the peaks found by the SNOW algorithm

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction
    algorithm using marker-based watershed segmentation".  Physical Review
    E. (2017)

    [2] Khan, ZA et al. "Dual network extraction algorithm to investigate
    multiple transport processes in porous materials: Image-based modeling
    of pore and grain-scale processes". Computers in Chemical Engineering.
    (2019)

    See Also
    --------
    snow_partitioning

    Notes
    -----
    In principle it is possible to perform a distance transform on each
    phase separately, merge these into a single image, then apply the
    watershed only once. This, however, has been found to create edge
    artifacts between regions arising from the way watershed handles
    plateaus in the distance transform. To overcome this, this function
    applies the watershed to each of the distance transforms separately,
    then merges the segmented regions back into a single image.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/snow_partitioning_n.html>`__
    to view online example.

    """
    # Perform snow on each phase and merge all segmentation and dt together
    phases_num = np.unique(im).astype(int)
    phases_num = phases_num[phases_num > 0]
    combined_dt = 0
    combined_region = 0
    _peaks = np.zeros_like(im, dtype=int)
    num = [0]
    for i, j in enumerate(phases_num):
        logger.info(f"Processing Phase {j}")
        # Isolate active phase from image
        phase = im == j
        # Limit peaks to active phase only
        temp = peaks*phase if peaks is not None else None
        phase_snow = snow_partitioning(phase, dt=None, r_max=r_max,
                                       sigma=sigma, peaks=temp)
        combined_dt += phase_snow.dt
        phase_snow.regions *= phase_snow.im
        phase_snow.regions += num[i]
        phase_ws = phase_snow.regions * phase_snow.im
        phase_ws[phase_ws == num[i]] = 0
        combined_region += phase_ws
        _peaks = _peaks + phase_snow.peaks + (phase_snow.peaks > 0)*num[i]
        num.append(np.amax(combined_region))

    tup = Results()
    tup.im = im
    tup.dt = combined_dt
    tup.phase_max_label = num[1:]
    tup.regions = combined_region
    tup.peaks = _peaks
    return tup


def find_peaks(dt, r_max=4, strel=None, sigma=None, parallel_kw={"divs": 1}):
    r"""
    Finds local maxima in the distance transform

    Parameters
    ----------
    dt : ndarray
        The distance transform of the pore space.  This may be calculated
        and filtered using any means desired.
    r_max : scalar
        The radius of the spherical element used in the maximum filter.
        This controls the localness of any maxima. The default is 4 voxels.
    strel : ndarray
        Instead of supplying ``r_max``, this argument allows a custom
        structuring element allowing control over both size and shape.
    sigma : float or list of floats
        If given, then a gaussian filter is applied to the distance transform
        using this value for the kernel
        (i.e. ``scipy.ndimage.gaussian_filter(dt, sigma)``)
    parallel_kw : dict
        Dictionary containing the settings for parallelization by chunking. The
        optional settings include `divs` (scalar or list of scalars,
        default = [2, 2, 2]), `overlap` (scalar or list of scalars, optional),
        and `cores` (scalar, default is all available cores).

        `divs` is the number of times to divide the image for parallel
        processing. If `1` then parallel processing does not occur. `2` is
        equivalent to `[2, 2, 2]` for a 3D image. If a list is provided, each
        respective axis will be divided by its corresponding number in the
        list. For example, [2, 3, 4] will divide z, y, and x axis to 2, 3,
        and 4 respectively.

        `overlap` is the amount of overlap to include when dividing up the
        image. This value is controlled by the size (i.e. radius) of the
        structuring element and cannot be controlled in this function using
        parallel_kw!

        `cores` is the number of cores that will be used to parallel process all
        domains. If ``None`` then all cores will be used but user can specify
        any integer values to control the memory usage. Setting value to 1 will
        effectively process the chunks in serial to minimize memory usage.

    Returns
    -------
    image : ndarray
        An array of booleans with ``True`` values at the location of any
        local maxima.

    Notes
    -----
    It is also possible ot the ``peak_local_max`` function from the
    ``skimage.feature`` module as follows:

    ``peaks = peak_local_max(image=dt, min_distance=r, exclude_border=0,
    indices=False)``

    The *skimage* function automatically uses a square structuring element
    which is significantly faster than using a circular or spherical
    element.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/find_peaks.html>`__
    to view online example.

    """
    # parse out divs from parallel_kw, take from settings if not given!
    divs = parallel_kw.get("divs", settings.divs)
    cores = parallel_kw.get("cores", settings.ncores)
    _check_for_singleton_axes(dt)
    if strel is None:
        strel = ps_round(r=r_max, ndim=dt.ndim)
    if sigma is not None:
        dt = spim.gaussian_filter(dt, sigma=sigma)
    parallel = False
    if isinstance(divs, int):
        divs = [divs]*len(dt.shape)
    if np.any(np.array(divs) > 1):
        parallel = True
        logger.info(f'Performing {inspect.currentframe().f_code.co_name} in parallel')
    if parallel:
        overlap = max(strel.shape)
        parallel_kw = {"divs": divs, "overlap": overlap, "cores": cores}
        mx = chunked_func(func=spim.maximum_filter, parallel_kw=parallel_kw,
                          im_arg='input', input=dt,
                          footprint=strel)
    else:
        mx = spim.maximum_filter(dt, footprint=strel)
    peaks = (dt == mx)
    return peaks


def reduce_peaks(peaks):
    r"""
    Any peaks that are broad or elongated are replaced with a single voxel
    that is located at the center of mass of the original voxels.

    Parameters
    ----------
    peaks : ndarray
        An image containing ``True`` values indicating peaks in the
        distance transform

    Returns
    -------
    image : ndarray
        An array with the same number of isolated peaks as the original
        image, but fewer total ``True`` voxels.

    Notes
    -----
    The center of mass of a group of voxels is used as the new single
    voxel, so if the group has an odd shape (like a horse shoe), the new
    voxel may *not* lie on top of the original set.

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/reduce_peaks.html>`__
    to view online example.

    """
    strel = footprint_rectangle((3,) * peaks.ndim)
    markers, N = spim.label(input=peaks, structure=strel)
    inds = spim.center_of_mass(
        input=peaks, labels=markers, index=np.arange(1, N + 1)
    )
    inds = np.floor(inds).astype(int)
    # Centroid may not be on old pixel, so create a new peaks image
    peaks_new = np.zeros_like(peaks, dtype=bool)
    peaks_new[tuple(inds.T)] = True
    return peaks_new


def trim_saddle_points(peaks, dt, maxiter=20):
    r"""
    Removes peaks that were mistakenly identified because they lied on a
    saddle or ridge in the distance transform that was not actually a true
    local peak.

    Parameters
    ----------
    peaks : ndarray
        A boolean image containing ``True`` values to mark peaks in the
        distance transform (``dt``)
    dt : ndarray
        The distance transform of the pore space for which the peaks
        are sought.
    maxiter : int
        The number of iteration to use when finding saddle points.
        The default value is 20.

    Returns
    -------
    image : ndarray
        An image with fewer peaks than the input image

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmentation".  Physical Review E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_saddle_points.html>`__
    to view online example.

    """
    new_peaks = np.zeros_like(peaks, dtype=bool)
    strel = footprint_rectangle((3,) * dt.ndim)
    labels, N = spim.label(peaks > 0)
    slices = spim.find_objects(labels)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, s in tqdm(enumerate(slices), desc=desc, **settings.tqdm):
        sx = extend_slice(s, shape=peaks.shape, pad=maxiter)
        peaks_i = labels[sx] == i + 1
        dt_i = dt[sx]
        im_i = dt_i > 0
        iters = 0
        while iters < maxiter:
            iters += 1
            peaks_dil = spim.binary_dilation(input=peaks_i, structure=strel)
            peaks_max = peaks_dil * np.amax(dt_i * peaks_dil)
            peaks_extended = (peaks_max == dt_i) * im_i
            if np.all(peaks_extended == peaks_i):
                new_peaks[sx] += peaks_i
                break  # Found a true peak
            elif np.sum(peaks_extended * peaks_i, dtype=np.int64) == 0:
                break  # Found a saddle point
            peaks_i = peaks_extended
        if iters >= maxiter:
            logger.debug(
                "Maximum number of iterations reached, consider "
                + "running again with a larger value of max_iters"
            )
    return new_peaks*peaks


def trim_saddle_points_legacy(peaks, dt, maxiter=10):
    r"""
    Removes peaks that were mistakenly identified because they lied on a
    saddle or ridge in the distance transform that was not actually a true
    local peak.

    Parameters
    ----------
    peaks : ND-array
        A boolean image containing True values to mark peaks in the distance
        transform (``dt``)
    dt : ND-array
        The distance transform of the pore space for which the true peaks are
        sought.
    maxiter : int
        The maximum number of iterations to run while eroding the saddle
        points.  The default is 10, which is usually not reached; however,
        a warning is issued if the loop ends prior to removing all saddle
        points.

    Returns
    -------
    image : ND-array
        An image with fewer peaks than the input image

    Notes
    -----
    This version of the function was included in versions of PoreSpy < 2. It
    is too aggressive in trimming peaks, so was rewritten for PoreSpy >= 2.
    This is included here for legacy reasons

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction algorithm
    using marker-based watershed segmenation".  Physical Review E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_saddle_points_legacy.html>`__
    to view online example.
    """
    new_peaks = np.zeros_like(peaks, dtype=bool)
    strel = footprint_rectangle((3,) * dt.ndim)
    labels, N = spim.label(peaks > 0)
    slices = spim.find_objects(labels)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i, s in tqdm(enumerate(slices), desc=desc, **settings.tqdm):
        sx = extend_slice(s, shape=peaks.shape, pad=10)
        peaks_i = labels[sx] == i + 1
        dt_i = dt[sx]
        im_i = dt_i > 0
        iters = 0
        while iters < maxiter:
            iters += 1
            peaks_dil = spim.binary_dilation(input=peaks_i, structure=strel)
            peaks_max = peaks_dil * np.amax(dt_i * peaks_dil)
            peaks_extended = (peaks_max == dt_i) * im_i
            if np.all(peaks_extended == peaks_i):
                break  # Found a true peak
            elif np.sum(peaks_extended * peaks_i, dtype=np.int64) == 0:
                peaks_i = False
                break  # Found a saddle point
            # The following line was also missing from the original.  It has
            # no effect on the result but without this the "maxiters" is
            # reached very often.
            peaks_i = peaks_extended
        # The following line is essentially a bug.  It should be:
        # peaks[s] += peaks_i. Without the += the peaks_i image overwrites
        # the entire slice s, which may include other peaks that are within
        # 10 voxels due to the padding of s with extend_slice.
        new_peaks[sx] = peaks_i
        if iters >= maxiter:
            logger.debug(
                "Maximum number of iterations reached, consider "
                + "running again with a larger value of max_iters"
            )
    return new_peaks*peaks


def trim_nearby_peaks(peaks, dt, f=1):
    r"""
    Removes peaks that are nearer to another peak than to solid

    Parameters
    ----------
    peaks : ndarray
        A image containing nonzeros values indicating peaks in the distance
        transform (``dt``).  If ``peaks`` is boolean, a boolean is returned;
        if ``peaks`` have already been labelled, then the original labels
        are returned, missing the trimmed peaks.
    dt : ndarray
        The distance transform of the pore space
    f : scalar
        Controls how close peaks must be before they are considered near
        to each other. Sets of peaks are tagged as too near if
        ``d_neighbor < f * d_solid``.

    Returns
    -------
    image : ndarray
        An array the same size and type as ``peaks`` containing a subset of
        the peaks in the original image.

    Notes
    -----
    Each pair of peaks is considered simultaneously, so for a triplet of nearby
    peaks, each pair is considered.  This ensures that only the single peak
    that is furthest from the solid is kept.  No iteration is required.

    References
    ----------
    [1] Gostick, J. "A versatile and efficient network extraction
    algorithm using marker-based watershed segmentation". Physical Review
    E. (2017)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/filters/reference/trim_nearby_peaks.html>`__
    to view online example.

    """
    strel = footprint_rectangle((3,) * dt.ndim)
    labels, N = spim.label(peaks > 0, structure=strel)
    crds = spim.center_of_mass(peaks > 0, labels=labels, index=np.arange(1, N + 1))
    try:
        crds = np.vstack(crds).astype(int)  # Convert to numpy array of ints
    except ValueError:
        return peaks
    L = dt[tuple(crds.T)]  # Get distance to solid for each peak
    drop_peaks = _nearby_peaks_to_drop(crds, L, f=f)

    new_peaks = ~np.isin(labels, drop_peaks+1)*peaks
    return new_peaks


def _estimate_overlap(im, mode='dt', zoom=0.25, dt=None):
    logger.info('Calculating overlap thickness')
    if mode == 'watershed':
        rev = spim.interpolation.zoom(im, zoom=zoom, order=0)
        rev = rev > 0
        if dt is None:
            dt = edt(rev)
        rev_snow = snow_partitioning(rev, dt=dt)
        labels, counts = np.unique(rev_snow, return_counts=True)
        node = np.where(counts == counts[1:].max())[0][0]
        slices = spim.find_objects(rev_snow)
        overlap = max(rev_snow[slices[node - 1]].shape) / (zoom * 2.0)
    if mode == 'dt':
        if dt is None:
            dt = edt((im > 0))
        overlap = dt.max()
    return overlap


def snow_partitioning_parallel(im,
                               r_max=4,
                               sigma=0.4,
                               parallel_kw={},
                               divs=2,
                               overlap=None,
                               cores=None,
                               dt=None,
                               maxiter=20,
                               mask_watershed=False,
                               ):
    r"""
    Performs the SNOW algorithm by geometric domain decomposition, with
    globally consistent watershed markers.

    The pipeline is split so that every step that decides *which markers
    exist* is exact (independent of ``divs``), while the two expensive
    volumetric steps stay chunked:

    1. Distance transform: computed once on the full image (or supplied).
    2. Candidate peaks (chunked, exact): Gaussian blur + maximum filter on
       each chunk with an overlap of ``blur_radius + r_max``. At that
       overlap both filters are local, so the result is identical to the
       global computation.
    3. Saddle trimming (parallel over peak clusters, exact): each
       face-connected peak cluster is tested against the *global* distance
       transform, exactly as ``trim_saddle_points`` does. Clusters are
       independent, so this parallelizes without any overlap.
    4. Nearby-peak trimming and marker labelling (global, sparse): done on
       peak coordinates only (KD-tree), so it is cheap even for large
       volumes. Markers are labelled once, globally, in raster order.
    5. Watershed (chunked): each chunk floods ``-dt`` with the global
       markers that fall inside its window (variable overlap, as before)
       and keeps only its core. Labels are already global, so there is no
       relabelling or stitching step.

    Only step 5 can differ from serial ``snow_partitioning``, and only where
    a basin extends beyond a chunk's window. Chunks have near-equal sizes,
    so the image is never padded or cropped.

    Parameters
    ----------
    im : ndarray
        Boolean image, ``True`` for the phase of interest.
    r_max : int
        Radius of the maximum filter used to find peaks.
    sigma : float
        Standard deviation of the Gaussian blur applied to ``dt`` before
        peak finding. 0 disables it.
    parallel_kw : dict
        ``divs`` (scalar or list), ``cores`` (int) and ``overlap`` (int,
        optional). ``overlap`` is a *minimum* watershed overlap in voxels
        per face; the variable overlap derived from the local maximum radius
        is used when larger. Keys in ``parallel_kw`` take precedence over
        the keyword arguments of the same name.
    dt : ndarray, optional
        Distance transform of ``im``. Computed if not given.
    maxiter : int
        Passed to the saddle-point trimming (same meaning as in
        ``trim_saddle_points``).
    mask_watershed : bool
        Flood only the phase of interest (see ``snow_partitioning``).

    Returns
    -------
    Results object with ``im``, ``dt``, ``regions`` and ``markers`` (a
    tuple ``(coords, labels)`` with the marker voxels in sparse form).
    """
    ndim = im.ndim
    divs = parallel_kw.get("divs", divs if divs is not None else settings.divs)
    cores = parallel_kw.get("cores", cores if cores is not None else settings.ncores)
    min_overlap = parallel_kw.get("overlap", overlap if overlap is not None else settings.overlap)
    divs = _normalize_divs(divs, im.shape)

    im = im > 0
    if dt is None:
        logger.info("Calculating distance transform")
        dt = edt(im)
    dt = np.ascontiguousarray(dt)
    if dt.shape != im.shape:
        raise ValueError(f"dt shape {dt.shape} does not match image shape {im.shape}")

    chunks = list(_chunk_grid(im.shape, divs))
    blur_r = int(4.0 * float(sigma) + 0.5) if sigma > 0 else 0
    marker_overlap = blur_r + int(np.ceil(r_max))

    executor = _ChunkExecutor(dt, cores)
    try:
        # Step 2: candidate peaks, chunked with a small exact overlap
        logger.info("Step 1 of 4: Finding candidate peaks (chunked)...")
        tasks = []
        for _, core in chunks:
            win, core_in_win = _window(core, [marker_overlap] * ndim,
                                       [marker_overlap] * ndim, im.shape)
            tasks.append((win, core_in_win, r_max, sigma))
        cand = executor.map(_task_candidate_peaks, tasks)
        cand = np.concatenate(cand, axis=0) if len(cand) else np.empty((0, ndim), np.int64)
        cand = _sort_raster(cand, im.shape)
        logger.info(f"Candidate peak voxels: {len(cand)}")

        # Step 3: saddle trimming, parallel over face-connected clusters
        logger.info("Step 2 of 4: Trimming saddle points (per cluster)...")
        comp = _connected_components(cand, connectivity=1)
        groups = _group_components(comp, executor.n_slots * 4)
        tasks = [(cand[vox_idx], comp_local, n_comp, maxiter)
                 for vox_idx, comp_local, n_comp in groups]
        kept = executor.map(_task_trim_saddle, tasks)
        kept = np.concatenate(kept) if len(kept) else np.empty(0, np.int64)
        if len(cand):
            cand_lin = np.ravel_multi_index(tuple(cand.T), im.shape)
            peaks = cand[np.isin(cand_lin, kept)]  # == new_peaks * peaks
        else:
            peaks = cand
        logger.info(f"Peak voxels after saddle trimming: {len(peaks)}")

        # Step 4: nearby-peak trimming and global labelling (sparse)
        logger.info("Step 3 of 4: Trimming nearby peaks and labelling markers...")
        marker_coords, marker_labels = _global_markers(peaks, dt, im.shape)
        n_markers = int(marker_labels.max()) if len(marker_labels) else 0
        logger.info(f"Number of markers: {n_markers}")

        # Step 5: chunked watershed with global markers
        logger.info("Step 4 of 4: Watershed (chunked, global markers)...")
        ws_overlaps = _watershed_overlaps(dt, chunks, divs, min_overlap)
        tasks = []
        for idx, core in chunks:
            lo, hi = ws_overlaps[idx]
            win, core_in_win = _window(core, lo, hi, im.shape)
            inside = np.ones(len(marker_coords), dtype=bool)
            for ax, w in enumerate(win):
                inside &= (marker_coords[:, ax] >= w.start) & (marker_coords[:, ax] < w.stop)
            start = np.array([w.start for w in win])
            tasks.append((win, core_in_win, marker_coords[inside] - start,
                          marker_labels[inside], mask_watershed))
        cores_out = executor.map(_task_watershed, tasks)
    finally:
        executor.close()

    regions = np.zeros(im.shape, dtype=np.int32)
    for (_, core), out in zip(chunks, cores_out):
        regions[core] = out

    tup = Results()
    tup.im = im
    tup.dt = dt
    tup.regions = regions
    tup.markers = (marker_coords, marker_labels)
    return tup


# ---------------------------------------------------------------------------
# Helpers for snow_partitioning_parallel
# ---------------------------------------------------------------------------

def _normalize_divs(divs, shape):
    ndim = len(shape)
    if np.isscalar(divs):
        divs = [divs] * ndim
    divs = [int(d) for d in divs]
    if len(divs) != ndim:
        raise ValueError(f"divs must be a scalar or have {ndim} entries, got {divs}")
    # Never make more chunks than voxels along an axis
    return [min(max(1, d), n) for d, n in zip(divs, shape)]


def _chunk_bounds(shape, divs):
    r"""Near-equal chunk boundaries per axis (sizes differ by at most 1)."""
    return [(np.arange(d + 1) * n) // d for n, d in zip(shape, divs)]


def _chunk_grid(shape, divs):
    bounds = _chunk_bounds(shape, divs)
    for idx in np.ndindex(*divs):
        core = tuple(slice(int(b[i]), int(b[i + 1])) for b, i in zip(bounds, idx))
        yield idx, core


def _window(core, lo, hi, shape):
    win = tuple(slice(max(0, c.start - int(l)), min(n, c.stop + int(h)))
                for c, l, h, n in zip(core, lo, hi, shape))
    core_in_win = tuple(slice(c.start - w.start, c.stop - w.start)
                        for c, w in zip(core, win))
    return win, core_in_win


def _min_chunk_shape(shape, divs):
    divs = _normalize_divs(divs, shape)
    return np.array([np.diff(b).min() for b in _chunk_bounds(shape, divs)])


def _sort_raster(coords, shape):
    if len(coords) == 0:
        return coords.reshape(0, len(shape)).astype(np.int64)
    flat = np.ravel_multi_index(tuple(coords.T), shape)
    return coords[np.argsort(flat, kind="stable")].astype(np.int64)


def _connected_components(coords, connectivity):
    r"""
    Connected components of a sparse voxel set, in raster order of each
    component's first voxel (the same order ``scipy.ndimage.label`` uses).

    connectivity=1 -> face neighbours (ndimage.label default)
    connectivity=ndim -> full neighbourhood (ps_rect(3, ndim))
    ``coords`` must already be sorted in raster order.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(coords)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    tree = sptl.cKDTree(coords)
    if connectivity == 1:
        pairs = tree.query_pairs(r=1, p=1, output_type="ndarray")
    else:
        pairs = tree.query_pairs(r=1, p=np.inf, output_type="ndarray")
    adj = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                     shape=(n, n))
    _, comp = connected_components(adj, directed=False)
    # Renumber components by first appearance (coords are raster-sorted)
    _, first = np.unique(comp, return_index=True)
    order = np.argsort(first)
    remap = np.empty_like(order)
    remap[order] = np.arange(len(order))
    return remap[comp]


def _group_components(comp, n_groups):
    r"""Round-robin assignment of clusters to groups (load balancing)."""
    n_comp = int(comp.max()) + 1 if len(comp) else 0
    if n_comp == 0:
        return []
    n_groups = max(1, min(n_groups, n_comp))
    group_of_comp = np.arange(n_comp) % n_groups
    vox_group = group_of_comp[comp]
    groups = []
    for g in range(n_groups):
        vox_idx = np.nonzero(vox_group == g)[0]
        uniq, local = np.unique(comp[vox_idx], return_inverse=True)
        groups.append((vox_idx, local, len(uniq)))
    return groups


def _nearby_peaks_to_drop(crds, L, f=1):
    r"""
    Core of ``trim_nearby_peaks`` on sparse peak coordinates.

    ``crds`` are integer peak coordinates (one per peak cluster) in raster
    order, ``L`` the distance to solid at each. Returns indices of peaks to
    drop. Ties in ``L`` are broken by position in ``crds`` (later wins),
    which is independent of the number of peaks. (The previous additive
    jiggle ``L + arange(N)*1e-6`` reached ~1 voxel for ~10^6 peaks and
    could override real differences in ``L``.)
    """
    n = len(crds)
    if n < 2:
        return np.empty(0, dtype=np.int64)
    tree = sptl.cKDTree(crds)
    dist, nn = tree.query(crds, k=2)
    nearest_neighbor = nn[:, 1]
    dist_to_neighbor = dist[:, 1]
    order = np.lexsort((np.arange(n), L))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    hits = np.nonzero(dist_to_neighbor <= f * L)[0]
    nbr = nearest_neighbor[hits]
    drop = np.where(rank[hits] < rank[nbr], hits, nbr)
    return np.unique(drop)


def _global_markers(peaks, dt, shape, f=1):
    r"""Sparse equivalent of trim_nearby_peaks + ndimage.label(ps_rect(3))."""
    ndim = len(shape)
    if len(peaks) == 0:
        return np.empty((0, ndim), dtype=np.int64), np.empty(0, dtype=np.int32)
    comp = _connected_components(peaks, connectivity=ndim)
    n_comp = int(comp.max()) + 1
    counts = np.bincount(comp, minlength=n_comp).astype(np.float64)
    com = np.stack([np.bincount(comp, weights=peaks[:, ax].astype(np.float64),
                                minlength=n_comp) / counts for ax in range(ndim)], axis=1)
    crds = com.astype(int)
    L = dt[tuple(crds.T)]
    drop = _nearby_peaks_to_drop(crds, L, f=f)
    keep = np.ones(n_comp, dtype=bool)
    keep[drop] = False
    new_label = np.cumsum(keep).astype(np.int32)  # raster order, 1..K
    vox_keep = keep[comp]
    return peaks[vox_keep], new_label[comp[vox_keep]]


def _watershed_overlaps(dt, chunks, divs, min_overlap=None):
    r"""
    Variable per-face watershed overlap, same rule as before:
    ceil(2 * max radius) over the chunk and its neighbour, where the radius
    map is max-filtered over the 3x3x3 chunk neighbourhood. Faces on the
    image boundary get no overlap.
    """
    ndim = dt.ndim
    max_r = np.zeros(divs, dtype=np.float64)
    for idx, core in chunks:
        sub = dt[core]
        max_r[idx] = sub.max() if sub.size else 0
    max_r = spim.maximum_filter(max_r, size=3, mode="constant", cval=0)
    min_overlap = int(min_overlap) if min_overlap else 0
    out = {}
    for idx, _ in chunks:
        lo = np.zeros(ndim, dtype=np.int64)
        hi = np.zeros(ndim, dtype=np.int64)
        for ax in range(ndim):
            for side, arr in ((-1, lo), (1, hi)):
                nb = list(idx)
                nb[ax] += side
                if 0 <= nb[ax] < divs[ax]:
                    r = max(max_r[idx], max_r[tuple(nb)])
                    # Cap at the image extent: an image without any solid
                    # has an infinite distance transform
                    ov = 2 * r
                    ov = int(np.ceil(ov)) if np.isfinite(ov) else dt.shape[ax]
                    arr[ax] = min(max(ov, min_overlap), dt.shape[ax])
        out[idx] = (lo, hi)
    return out


class _ChunkExecutor:
    r"""
    Runs chunk tasks on threads (local) or on a dask.distributed client
    (e.g. SLURM). In distributed mode the distance transform is written
    once to a shared .npy file and workers memory-map only what they need.
    """

    def __init__(self, dt, cores, directory="regions_npy_stack"):
        self.cores = cores if cores else (os.cpu_count() or 1)
        self.path = None
        try:
            self.client = dask.distributed.get_client()
        except ValueError:
            self.client = None
        if self.client is not None:
            self.n_slots = max(1, sum(self.client.nthreads().values()))
        else:
            self.n_slots = self.cores
        if self.client is not None:
            d = Path(os.getcwd()) / directory
            os.makedirs(d, exist_ok=True)
            self.path = d / "dt.npy"
            np.save(self.path, dt)
            self.src = str(self.path)
        else:
            self.src = dt

    def map(self, fn, arg_tuples):
        if self.client is not None:
            futures = [self.client.submit(fn, self.src, *args, pure=False)
                       for args in arg_tuples]
            return self.client.gather(futures)
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.cores) as ex:
            return list(ex.map(lambda args: fn(self.src, *args), arg_tuples))

    def close(self):
        if self.path is not None:
            try:
                os.remove(self.path)
            except OSError:
                pass


def _open_dt(src):
    return np.load(src, mmap_mode="r") if isinstance(src, str) else src


def _as_3d(arr):
    return arr.reshape((1,) * (3 - arr.ndim) + arr.shape)


def _task_candidate_peaks(src, win, core_in_win, r_max, sigma):
    dt = np.asarray(_open_dt(src)[win])
    ndim = dt.ndim
    mask = dt > 0
    if sigma > 0:
        dt_blurred = spim.gaussian_filter(input=dt, sigma=sigma) * mask
    else:
        dt_blurred = np.copy(dt)
    # Same footprint as find_peaks; peaks restricted to pore voxels (solid
    # "peaks" are always removed by the saddle trimming anyway)
    strel = ps_round(r=r_max, ndim=ndim)
    offsets = np.argwhere(strel) - np.array(strel.shape) // 2
    offsets = offsets[np.any(offsets != 0, axis=1)]
    offsets = offsets[np.argsort((offsets**2).sum(axis=1), kind="stable")]
    pad = 3 - ndim
    offsets = np.hstack([np.zeros((len(offsets), pad), dtype=np.int64),
                         offsets.astype(np.int64)])
    c0 = np.array([0] * pad + [c.start for c in core_in_win], dtype=np.int64)
    c1 = np.array([1] * pad + [c.stop for c in core_in_win], dtype=np.int64)
    peaks = _nb_peaks_in_core(np.ascontiguousarray(_as_3d(dt_blurred)), offsets, c0, c1)
    crd = np.argwhere(peaks.reshape(peaks.shape[pad:]))
    crd += np.array([w.start + c.start for w, c in zip(win, core_in_win)])
    return crd.astype(np.int64)


def _task_trim_saddle(src, coords, comp_local, n_comp, maxiter):
    r"""Runs the numba saddle kernel on a group of clusters.
    Returns global linear indices of kept voxels (before intersection
    with the candidate set)."""
    dt = _open_dt(src)
    if isinstance(dt, np.memmap):
        dt = dt.view(np.ndarray)
    ndim = dt.ndim
    order = np.argsort(comp_local, kind="stable")
    ptr = np.zeros(n_comp + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(comp_local, minlength=n_comp))
    crd = coords[order].astype(np.int64)
    if ndim < 3:
        crd = np.hstack([np.zeros((len(crd), 3 - ndim), dtype=np.int64), crd])
    # Linear indices are identical for the 3D view of a 2D array
    return _nb_trim_saddle(_as_3d(dt), crd, ptr, int(maxiter))


def _task_watershed(src, win, core_in_win, marker_coords, marker_labels,
                    mask_watershed=False):
    dt = np.asarray(_open_dt(src)[win])
    mask = dt > 0
    if len(marker_labels) == 0:
        return mask[core_in_win].astype(np.int32)
    markers = np.zeros(dt.shape, dtype=np.int32)
    markers[tuple(marker_coords.T)] = marker_labels
    regions = watershed(image=-dt, markers=markers,
                        mask=mask if mask_watershed else None)
    return (regions[core_in_win] * mask[core_in_win]).astype(np.int32)


@nb.njit(cache=True)
def _get_chunks_overlaps(max_radii_map):
    shape = max_radii_map.shape
    ndim = max_radii_map.ndim
    chunks_overlaps = np.empty(shape + (ndim, 2), dtype=np.int64)

    neighbor_directions_idxs = [
        [[(-1, 0, 0)], [(1, 0, 0)]],
        [[(0, -1, 0)], [(0, 1, 0)]],
        [[(0, 0, -1)], [(0, 0, 1)]],
    ]

    for chunk_idx in np.ndindex(shape):
        i, j, k = chunk_idx
        for axis_index in range(ndim):
            axis_directions = neighbor_directions_idxs[axis_index]
            for direction_index, neighbor_directions in enumerate(axis_directions):
                overlap = 0
                for (di, dj, dk) in neighbor_directions:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if (
                        0 <= ni < shape[0] and
                        0 <= nj < shape[1] and
                        0 <= nk < shape[2]
                    ):
                        neighbor_idx = (ni, nj, nk)
                        max_radius = max(max_radii_map[chunk_idx], max_radii_map[neighbor_idx])
                        overlap = max(overlap, int(np.ceil(2*max_radius)))
                chunks_overlaps[chunk_idx][axis_index, direction_index] = overlap

    return chunks_overlaps

def _get_chunk_with_overlap(dt_chunks, chunk_idx, chunk_shape, chunks_overlaps):
    """
    Extracts a chunk from the main image `dt` with the specified variable
    overlap on each face.
    """
    s_ = []
    for i in range(dt_chunks.ndim):
        start = chunk_idx[i] * chunk_shape[i]
        end = start + chunk_shape[i]
        s_start = start - chunks_overlaps[chunk_idx][i, 0]
        s_end = end + chunks_overlaps[chunk_idx][i, 1]
        s_.append(slice(max(0, s_start), min(dt_chunks.shape[i], s_end)))
    return dt_chunks[tuple(s_)]

def _trim_variable_overlap(chunk, chunk_idx, overlaps):
    """
    Trims the processed chunk to remove the variable overlap.
    """
    s_ = []
    for i in range(chunk.ndim):
        # Trimm but let 1 voxel border
        start = overlaps[chunk_idx][i, 0]
        end = chunk.shape[i] - overlaps[chunk_idx][i, 1]
        start -= int(overlaps[chunk_idx][i, 0] != 0)
        end += int(overlaps[chunk_idx][i, 1] != 0)
        s_.append(slice(start, end))
    return chunk[tuple(s_)]

def _compute_max_radii_numpy(dt, divs, chunk_shape):
    """
    Computes the maximum value within each chunk of a numpy array.

    This function is a numpy-only replacement for dask's:
    `da.from_array(dt, chunks=chunk_shape).map_blocks(np.max).compute()`

    Parameters
    ----------
    dt : ndarray
        The input numpy array (distance transform).
    divs : tuple or list
        The number of divisions along each axis.
    chunk_shape : tuple
        The shape of a single chunk.

    Returns
    -------
    ndarray
        An array where each element is the maximum value of the
        corresponding chunk in the input array. The shape of the array
        will be equal to `divs`.
    """
    max_radii = np.zeros(divs, dtype=dt.dtype)
    for index in np.ndindex(tuple(divs)):
        s_ = []
        for i in range(dt.ndim):
            start = index[i] * chunk_shape[i]
            end = start + chunk_shape[i]
            s_.append(slice(start, end))
        chunk = dt[tuple(s_)]
        if chunk.size > 0:
            max_radii[index] = np.max(chunk)
        else:
            max_radii[index] = 0
    return max_radii

@dask.delayed
def _process_chunk_delayed(chunk_with_overlap, chunk_idx, overlaps, r_max, sigma):
    """
    A dask.delayed function that processes a single numpy chunk.
    Dask computes the chunk with overlap and passes it here as a numpy array.
    """
    # Process the chunk using the core SNOW logic
    processed_chunk = _snow_chunked(chunk_with_overlap, r_max=r_max, sigma=sigma)

    # Trim the processed chunk to remove overlap
    trimmed_chunk = _trim_variable_overlap(processed_chunk, chunk_idx, overlaps)

    return trimmed_chunk, chunk_idx


def _save_to_stack_dask(dt, directory, chunk_shape):
    with dask.config.set(scheduler="single-threaded"):
        regions = da.from_array(dt, chunks=chunk_shape)
        os.makedirs(directory, exist_ok=True)
        def save_block(block, block_info=None):
            idx = block_info[None]["chunk-location"]
            filename = str(Path(directory) / f"{idx[0]}-{idx[1]}-{idx[2]}.npy")
            np.save(filename, block)
            return block
        regions.map_blocks(save_block, dtype=regions.dtype).compute()


def _save_to_stack_numpy(dt, directory, chunk_shape):
    """
    Splits a volume into chunks and saves them to npy files
    using only Numpy and standard libraries.
    """
    os.makedirs(directory, exist_ok=True)

    # Unpack shapes
    # Assuming 3D volume (z, y, x)
    d_z, d_y, d_x = dt.shape
    c_z, c_y, c_x = chunk_shape

    # Calculate how many chunks are needed in each dimension
    # math.ceil ensures we include the last partial chunk
    n_chunks_z = math.ceil(d_z / c_z)
    n_chunks_y = math.ceil(d_y / c_y)
    n_chunks_x = math.ceil(d_x / c_x)

    # Iterate through the grid of chunks
    for z in range(n_chunks_z):
        for y in range(n_chunks_y):
            for x in range(n_chunks_x):
                # Calculate slice start positions
                z_start = z * c_z
                y_start = y * c_y
                x_start = x * c_x

                # Calculate slice end positions
                # We use min() to handle the edge of the array
                # (e.g., if array is 100 wide and chunk is 64, end is 100, not 128)
                z_end = min(z_start + c_z, d_z)
                y_end = min(y_start + c_y, d_y)
                x_end = min(x_start + c_x, d_x)

                # Slice the numpy array
                block = dt[z_start:z_end, y_start:y_end, x_start:x_end]

                # Construct filename: z-y-x.npy
                filename = Path(directory) / f"{z}-{y}-{x}.npy"

                # Save
                np.save(filename, block)


def _load_delayed_from_stack(directory_name, chunk_shape, divs, dtype):
    directory_path = Path(os.getcwd()) / directory_name
    files = os.listdir(str(directory_path))
    def parse_filename(fname):
        name = os.path.splitext(fname)[0]  # remove .npy
        return tuple(map(int, name.split('-')))
    blocks = {}
    for f in files:
        idx = parse_filename(f)
        path = str(directory_path / f)
        blocks[idx] = dask.delayed(np.load)(path)
    nested_blocks = [[[blocks[(i,j,k)] for k in range(divs[2])]
                for j in range(divs[1])]
                for i in range(divs[0])]
    dask_blocks = [[[da.from_delayed(block, shape=chunk_shape, dtype=dtype)
                for block in row] for row in plane] for plane in nested_blocks]
    regions = da.block(dask_blocks)
    return regions


def _pad(im, pad_width=1, constant_value=0):
    r"""
    Pad the image with a constant values and width.

    Parameters
    ----------
    im : ndarray
        The image that requires padding
    pad_width : int
        The number of values that will be padded from the edges. Default
        values is 1.
    constant_value : int
        Pads with the specified constant value

    Returns
    -------
    output: ndarray
        Padded image with same dimensions as provided image

    """
    shape = np.array(im.shape)
    pad_shape = shape + (2 * pad_width)
    temp = np.zeros(pad_shape, dtype=np.uint32)
    if constant_value != 0:
        temp = temp + constant_value
    if im.ndim == 3:
        temp[pad_width: -pad_width,
             pad_width: -pad_width,
             pad_width: -pad_width] = im
    elif im.ndim == 2:
        temp[pad_width: -pad_width,
             pad_width: -pad_width] = im
    else:
        temp[pad_width: -pad_width] = im

    return temp


def relabel_chunks(im, chunk_shape):
    r"""
    Assign new labels to each chunk or sub-domain of actual image. This
    prevents from two or more regions to have same label.

    Parameters
    ----------
    im: ndarray
        Actual image that contains repeating labels in chunks/sub-domains.
    chunk_shape: tuple
        The shape of chunk that will be relabeled in actual image. Note
        the chunk shape should be a multiple of actual image shape
        otherwise some labels will not be relabeled.

    Returns
    -------
    output : ndarray
        Relabeled image with unique label assigned to each region.

    """
    im = _pad(im, pad_width=1)
    im_shape = np.array(im.shape, dtype=np.uint32)
    max_num = 0
    c = np.array(chunk_shape, dtype=np.uint32) + 2
    num = (im_shape / c).astype(int)

    if im.ndim == 3:
        for z in range(num[0]):
            for y in range(num[1]):
                for x in range(num[2]):
                    chunk = im[z * c[0]: (z + 1) * c[0],
                               y * c[1]: (y + 1) * c[1],
                               x * c[2]: (x + 1) * c[2]]
                    chunk += max_num
                    chunk[chunk == max_num] = 0
                    max_num = chunk.max()
                    im[z * c[0]: (z + 1) * c[0],
                       y * c[1]: (y + 1) * c[1],
                       x * c[2]: (x + 1) * c[2]] = chunk
    else:
        for y in range(num[0]):
            for x in range(num[1]):
                chunk = im[y * c[0]: (y + 1) * c[0],
                           x * c[1]: (x + 1) * c[1]]
                chunk += max_num
                chunk[chunk == max_num] = 0
                max_num = chunk.max()
                im[y * c[0]: (y + 1) * c[0],
                   x * c[1]: (x + 1) * c[1]] = chunk

    return im


def _trim_internal_slice(im, chunk_shape):
    r"""
    Delete extra slices from image that were used to stitch two or more
    chunks together.

    Parameters:
    -----------
    im :  ndarray
        image that contains extra slices in x, y, z direction.
    chunk_shape : tuple
        The shape of the chunk from which image is subdivided.

    Return:
    -------
    output : ndarray
        Image without extra internal slices. The shape of the image will
        be same as input image provided for watershed segmentation.

    """
    im_shape = np.array(im.shape, dtype=np.uint32)
    c1 = np.array(chunk_shape, dtype=np.uint32) + 2
    c2 = np.array(chunk_shape, dtype=np.uint32)
    num = (im_shape / c1).astype(int)
    out_shape = num * c2
    out = np.empty((out_shape), dtype=np.uint32)

    if im.ndim == 3:
        for z in range(num[0]):
            for y in range(num[1]):
                for x in range(num[2]):
                    chunk = im[z * c1[0]: (z + 1) * c1[0],
                               y * c1[1]: (y + 1) * c1[1],
                               x * c1[2]: (x + 1) * c1[2]]

                    out[z * c2[0]: (z + 1) * c2[0],
                        y * c2[1]: (y + 1) * c2[1],
                        x * c2[2]: (x + 1) * c2[2]] = chunk[1:-1, 1:-1, 1:-1]
    else:
        for y in range(num[0]):
            for x in range(num[1]):
                chunk = im[y * c1[0]: (y + 1) * c1[0],
                           x * c1[1]: (x + 1) * c1[1]]

                out[y * c2[0]: (y + 1) * c2[0],
                    x * c2[1]: (x + 1) * c2[1]] = chunk[1:-1, 1:-1]

    return out


def _watershed_stitching(im, chunk_shape):
    r"""
    Stitch individual sub-domains of watershed segmentation into one big
    segmentation with all boundary labels of each sub-domain relabeled to
    merge boundary regions.

    Parameters:
    -----------
    im : ndarray
        A worked image with watershed segmentation performed on all
        sub-domains individually.
    chunk_shape: tuple
        The shape of the sub-domain in which segmentation is performed.

    Returns
    -------
    output : ndarray
        Stitched watershed segmentation with all sub-domains merged to
        form a single watershed segmentation.

    """
    c_shape = np.array(chunk_shape)
    cuts_num = (np.array(im.shape) / (c_shape + 2)).astype(np.uint32)

    for axis, num in enumerate(cuts_num):
        keys = []
        values = []
        if num > 1:
            im = im.swapaxes(0, axis)
            for i in range(1, num):
                sl = i * (chunk_shape[axis] + 2)
                sl1 = im[sl - 2, ...]
                sl2 = im[sl - 0, ...]
                sl1_labels = sl1.flatten()[sl1.flatten() > 0]
                sl2_labels = sl2.flatten()[sl2.flatten() > 0]
                if sl1_labels.size != sl2_labels.size:
                    raise Exception('The selected overlapping thickness is not '
                                    'suitable for input image. Change '
                                    'overlapping criteria '
                                    'or manually input value.')
                keys.append(sl2_labels)
                values.append(sl1_labels)
            im = _replace_labels(array=im, keys=keys, values=values)
            im = im.swapaxes(axis, 0)
    im = _trim_internal_slice(im=im, chunk_shape=chunk_shape)
    im = _resequence_labels(array=im)

    return im


_KEY_VALUE_DICT_TYPE = nb.types.DictType(nb.types.int64, nb.types.int64)

@nb.njit(cache=False)  # Can't use cache because of _translate
def _replace_labels(array, keys, values):
    r"""
    Replace labels in array provided as keys to values.

    Parameter:
    ----------
    array : ndarray
        Array which requires replacing labels
    keys :  1D-array
        The unique labels that need to be replaced
    values : 1D-array
        The unique values that will be assigned to labels

    return:
    -------
    array : ndarray
        Array with replaced labels.
    """
    key_values_dict = nb.typed.Dict.empty(
        key_type=nb.types.int64,
        value_type=_KEY_VALUE_DICT_TYPE,
    )
    for i in range(len(values)):
        for j in range(len(values[i])):
            value = values[i][j]
            key = keys[i][j]

            if key not in key_values_dict:
                value_count_dict = nb.typed.Dict.empty(
                    key_type=nb.types.int64,
                    value_type=nb.types.int64,
                )
                key_values_dict[key] = value_count_dict
                value_count_dict[value] = 1
            else:
                value_count_dict = key_values_dict[key]

                if value not in value_count_dict:
                    value_count_dict[value] = 1
                else:
                    value_count_dict[value] += 1

    translation_dict = nb.typed.Dict.empty(
        key_type=nb.types.int64,
        value_type=nb.types.int64,
    )
    for key, value_count_dict in key_values_dict.items():
        target_value = 0
        max_count = 0
        for value, count in value_count_dict.items():
            if count > max_count:
                target_value = value
                max_count = count

        if max_count != 0:
            translation_dict[key] = target_value

    _translate(array, translation_dict)
    return array


@nb.njit(cache=False)  # Can't use cache for recursive functions
def _translate(array, translation_dict):
    if array.ndim > 1:
        for i in range(array.shape[0]):
            new_array = array[i]
            _translate(new_array, translation_dict)
    else:
        for i in range(array.shape[0]):
            old_value = array[i]
            if old_value in translation_dict:
                array[i] = translation_dict[old_value]


@nb.njit()
def _sequence(array, count):
    r"""
    Internal function of resequnce_labels method. This function resquence
    array elements in an ascending order using numba technique which is
    many folds faster than make contigious funcition.

    Parameters
    ----------
    array: ndarray
        1d array that needs resquencing.
    count: array_like
        1d array of zeros having same size as array.

    Returns
    -------
    array: 1d-array
        The input array with elements resequenced in ascending order

    Notes
    -----
    The output of this function is not same as make_contigous or
    relabel_sequential function of scikit-image. This function resequence
    and randomize the regions while other methods only do resequencing and
    output sorted array.

    """
    a = 1
    i = 0
    while i < (len(array)):
        data = array[i]
        if data != 0:
            if count[data] == 0:
                count[data] = a
                a += 1
        array[i] = count[data]
        i += 1


@nb.njit(parallel=True)
def _amax(array):
    r"""
    Find largest element in an array using fast parallel numba technique.

    Parameter:
    ----------
    array: ndarray
        Array in which largest elements needs to be calcuted.

    Returns
    -------
    scalar: float or int
        The largest element value in the input array.

    """
    return np.max(array)


def _resequence_labels(array):
    r"""
    Resequence the lablels to make them contigious.

    Parameters
    ----------
    array: ndarray
        Array that requires resequencing.

    Returns
    -------
    array : ndarray
        Resequenced array with same shape as input array.

    """
    a_shape = array.shape
    array = array.ravel()
    max_num = _amax(array) + 1
    count = np.zeros(max_num, dtype=np.uint32)
    _sequence(array, count)

    return array.reshape(a_shape)


def _snow_chunked(dt, r_max=5, sigma=0.4):
    r"""
    This private version of snow is called during snow_parallel.
    """
    mask = dt > 0
    dt_blurred = spim.gaussian_filter(input=dt, sigma=sigma)*mask
    peaks = find_peaks(dt=dt_blurred, r_max=r_max)
    peaks = trim_saddle_points(peaks=peaks, dt=dt)
    if len(peaks) > 0:
        peaks = trim_nearby_peaks(peaks=peaks, dt=dt)
        peaks, N = spim.label(peaks > 0, structure=ps_rect(3, dt.ndim))
        regions = watershed(image=-dt, markers=peaks)
    else:
        regions = np.ones_like(dt_blurred)
    return regions * mask


class Divisible:
    def __init__(self, reference_im, divs):
        self.divs = np.array(divs)
        self.ndim = reference_im.ndim
        self.original_shape = np.array(reference_im.shape)

        self.new_shape = []
        for i in range(reference_im.ndim):
            self.new_shape.append(int(np.ceil(self.original_shape[i] / divs[i]) * divs[i]))
        self.new_shape = np.array(self.new_shape)

        self.padding = self.new_shape - self.original_shape

        self.chunk_shape = self.new_shape // self.divs

    def get_shape(self):
        return self.new_shape

    def get_chunk_shape(self):
        return self.chunk_shape

    def make_divisible(self, im):
        return np.pad(im, [(0, p) for p in self.padding], mode='edge')

    def make_original_shape(self, im):
        return im[tuple(slice(0, s) for s in self.original_shape)]


# ---------------------------------------------------------------------------
# Numba kernels (nogil, serial). Parallelism comes from running several
# kernel calls concurrently on threads (they release the GIL) or on dask
# workers. parallel=True is deliberately avoided: concurrent calls into
# numba's default "workqueue" threading layer from several Python threads
# or dask worker threads are not thread-safe.
# ---------------------------------------------------------------------------

@nb.njit(nogil=True, cache=True)
def _nb_peaks_in_core(blurred, offsets, c0, c1):
    r"""
    Peak test of ``find_peaks`` (``dt == maximum_filter(dt, footprint)``)
    for pore voxels (``blurred > 0``) inside the core ``[c0, c1)``.

    A voxel is a peak iff no voxel under the footprint is strictly larger.
    Offsets are sorted nearest-first so most voxels exit after a few reads.
    Out-of-bounds neighbours are skipped: under scipy's default 'reflect'
    mode a reflected neighbour of a ball footprint always lands inside the
    same footprint, so skipping gives the identical result.
    """
    Z, Y, X = blurred.shape
    out = np.zeros((c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]), dtype=np.bool_)
    n_off = offsets.shape[0]
    for z in range(c0[0], c1[0]):
        for y in range(c0[1], c1[1]):
            for x in range(c0[2], c1[2]):
                v = blurred[z, y, x]
                if v <= 0:
                    continue
                is_peak = True
                for k in range(n_off):
                    zz = z + offsets[k, 0]
                    yy = y + offsets[k, 1]
                    xx = x + offsets[k, 2]
                    if zz < 0 or zz >= Z or yy < 0 or yy >= Y or xx < 0 or xx >= X:
                        continue
                    if blurred[zz, yy, xx] > v:
                        is_peak = False
                        break
                out[z - c0[0], y - c0[1], x - c0[2]] = is_peak
    return out


@nb.njit(nogil=True, cache=True)
def _nb_trim_saddle(dt, coords, ptr, maxiter):
    r"""
    Per-cluster loop of ``trim_saddle_points`` on sparse voxel sets.

    ``coords`` holds the voxels of all clusters, grouped: cluster ``c`` is
    ``coords[ptr[c]:ptr[c+1]]``. Returns the global linear indices of the
    final ``peaks_i`` of every cluster classified as a true peak (the
    caller intersects them with the candidate set, i.e. ``new_peaks*peaks``).

    Instead of dilating a dense box each iteration, the current set is kept
    as a list and its 3x3x3 dilation is built from the neighbours of its
    voxels, deduplicated with stamp arrays (no clearing between clusters).
    The dilation can never leave ``bbox +/- maxiter``, which is exactly the
    ``extend_slice(..., pad=maxiter)`` box of the original, so the result is
    identical.
    """
    Z, Y, X = dt.shape
    n_cl = ptr.shape[0] - 1

    # Size the reusable buffers for the largest box
    max_box = 1
    for c in range(n_cl):
        lo0 = Z; lo1 = Y; lo2 = X; hi0 = 0; hi1 = 0; hi2 = 0
        for i in range(ptr[c], ptr[c + 1]):
            lo0 = min(lo0, coords[i, 0]); hi0 = max(hi0, coords[i, 0] + 1)
            lo1 = min(lo1, coords[i, 1]); hi1 = max(hi1, coords[i, 1] + 1)
            lo2 = min(lo2, coords[i, 2]); hi2 = max(hi2, coords[i, 2] + 1)
        b = ((min(Z, hi0 + maxiter) - max(0, lo0 - maxiter))
             * (min(Y, hi1 + maxiter) - max(0, lo1 - maxiter))
             * (min(X, hi2 + maxiter) - max(0, lo2 - maxiter)))
        max_box = max(max_box, b)

    dmark = np.zeros(max_box, dtype=np.int64)   # membership in dilation
    cmark = np.zeros(max_box, dtype=np.int64)   # membership in current set
    cur = np.empty(max_box, dtype=np.int64)
    nxt = np.empty(max_box, dtype=np.int64)
    dil = np.empty(max_box, dtype=np.int64)
    dstamp = 0
    cstamp = 0

    out = np.empty(max(16, coords.shape[0]), dtype=np.int64)
    n_out = 0

    for c in range(n_cl):
        lo0 = Z; lo1 = Y; lo2 = X; hi0 = 0; hi1 = 0; hi2 = 0
        for i in range(ptr[c], ptr[c + 1]):
            lo0 = min(lo0, coords[i, 0]); hi0 = max(hi0, coords[i, 0] + 1)
            lo1 = min(lo1, coords[i, 1]); hi1 = max(hi1, coords[i, 1] + 1)
            lo2 = min(lo2, coords[i, 2]); hi2 = max(hi2, coords[i, 2] + 1)
        b0 = max(0, lo0 - maxiter); e0 = min(Z, hi0 + maxiter)
        b1 = max(0, lo1 - maxiter); e1 = min(Y, hi1 + maxiter)
        b2 = max(0, lo2 - maxiter); e2 = min(X, hi2 + maxiter)
        s0 = e0 - b0; s1 = e1 - b1; s2 = e2 - b2

        cstamp += 1
        ncur = 0
        for i in range(ptr[c], ptr[c + 1]):
            u = ((coords[i, 0] - b0) * s1 + (coords[i, 1] - b1)) * s2 + (coords[i, 2] - b2)
            if cmark[u] != cstamp:
                cmark[u] = cstamp
                cur[ncur] = u
                ncur += 1

        for _ in range(maxiter):
            # 3x3x3 dilation of the current set, clipped to the box
            dstamp += 1
            ndil = 0
            for j in range(ncur):
                u = cur[j]
                z = u // (s1 * s2)
                y = (u // s2) % s1
                x = u % s2
                for dz in range(-1, 2):
                    zz = z + dz
                    if zz < 0 or zz >= s0:
                        continue
                    for dy in range(-1, 2):
                        yy = y + dy
                        if yy < 0 or yy >= s1:
                            continue
                        for dx in range(-1, 2):
                            xx = x + dx
                            if xx < 0 or xx >= s2:
                                continue
                            w = (zz * s1 + yy) * s2 + xx
                            if dmark[w] != dstamp:
                                dmark[w] = dstamp
                                dil[ndil] = w
                                ndil += 1
            # Max of dt over the dilation (dt >= 0, so 0 is the floor, as
            # in np.amax(dt_i * peaks_dil))
            m = dt[b0, b1, b2] * 0
            for j in range(ndil):
                w = dil[j]
                val = dt[b0 + w // (s1 * s2), b1 + (w // s2) % s1, b2 + w % s2]
                if val > m:
                    m = val
            # peaks_extended: voxels of the dilation equal to the max, in pore
            nnext = 0
            overlap = 0
            for j in range(ndil):
                w = dil[j]
                val = dt[b0 + w // (s1 * s2), b1 + (w // s2) % s1, b2 + w % s2]
                if val == m and val > 0:
                    nxt[nnext] = w
                    nnext += 1
                    if cmark[w] == cstamp:
                        overlap += 1
            if nnext == ncur and overlap == ncur:
                # True peak: emit the current set
                if n_out + ncur > out.shape[0]:
                    new = np.empty(2 * out.shape[0] + ncur, dtype=np.int64)
                    new[:n_out] = out[:n_out]
                    out = new
                for j in range(ncur):
                    u = cur[j]
                    gz = b0 + u // (s1 * s2)
                    gy = b1 + (u // s2) % s1
                    gx = b2 + u % s2
                    out[n_out] = (gz * Y + gy) * X + gx
                    n_out += 1
                break
            elif overlap == 0:
                break  # saddle point
            # Keep walking along the plateau
            tmp = cur
            cur = nxt
            nxt = tmp
            ncur = nnext
            cstamp += 1
            for j in range(ncur):
                cmark[cur[j]] = cstamp
        # maxiter reached without a decision: dropped (as in the original)
    return out[:n_out]
