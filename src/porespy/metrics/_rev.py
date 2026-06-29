import inspect
import logging
import time

import dask
import numpy as np
import pandas as pd

from porespy.tools import (
    Results,
    get_slices_grid,
    get_slices_random,
    get_tqdm,
    settings,
)

__all__ = [
    "rev_porosity",
    "rev_tortuosity",
]


logger = logging.getLogger()
tqdm = get_tqdm()


def rev_porosity(im, n=1000, slices=None):
    r"""
    Calculates the porosity for a many subdomains of diffrent sizes, suitable for
    an REV plot

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous material with `True` indicating the phase of
        interest
    n : int, default = 1000
        The number of random blocks to analyze. If `slices` are provided this is
        ignored.
    slices : list
        A list of `slice` objects into the image which define the subdomains. If
        not provided then `n` random blocks are used.

    Returns
    -------
    result : Results object
        A custom object with the following data added as named attributes:

        ========== ==================================================================
        Attribute  Description
        ========== ==================================================================
        volume     The total volume of each cubic subdomain tested
        porosity   The porosity of each subdomain tested
        ========== ==================================================================

        These attributes can be conveniently plotted by passing the Results
        object to matplotlib's ``plot`` function using the notation:
        ``plt.plot(\*result, 'b.')``. The resulting plot is similar to the
        sketch given by Bachmat and Bear [1]_

    References
    ----------
    .. [1] Bachmat and Bear. On the Concept and Size of a Representative
       Elementary Volume (Rev), Advances in Transport Phenomena in Porous Media
       (1987)

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/rev_porosity.html>`__
    to view online example.

    """
    # TODO: This function is frustratingly slow.  Profiling indicates that all the
    # time is spent on scipy's ``sum`` function which is needed to sum the number of
    # void voxels (1's) in each subdomain.

    # TODO: this function is a prime target for parallelization since the
    # `n` points are calculated independently.
    if slices is None:
        slices = get_slices_random(im=im, n=n)

    N = len(slices)
    porosity = np.zeros(shape=(N,), dtype=float)
    volume = np.zeros(shape=(N,), dtype=int)
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for i in tqdm(np.arange(0, N), desc=desc, **settings.tqdm):
        temp = im[slices[i]]
        Vp = np.sum(temp, dtype=np.int64)
        Vt = np.size(temp)
        porosity[i] = Vp / Vt
        volume[i] = Vt
    profile = Results()
    profile.volume = volume
    profile.porosity = porosity
    return profile


def calc_g(im, axis, solver_args={}):
    r"""
    Calculates diffusive conductance of an image in the direction specified

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    axis : int
        0 for x-axis, 1 for y-axis, 2 for z-axis.
    solver_args : dict
        Dictionary of keyword arguments to pass on to the solver.  The most
        relevant one being `'tol'` which is 1e-6 by default. Using larger values
        might improve speed at the cost of accuracy.

    Returns
    -------
    results : dataclass-like
        An object with the results of the calculation as attributes

    """
    from porespy.simulations import tortuosity_fd

    solver_args = {"tol": 1e-6} | solver_args
    solver = solver_args.pop("solver", None)
    t0 = time.perf_counter()

    try:
        results = tortuosity_fd(im=im, axis=axis, solver=solver)
    except Exception:
        results = Results()
        results.effective_porosity = 0.0
        results.original_porosity = im.sum() / im.size
        results.tortuosity = np.inf
        results.time = time.perf_counter() - t0
    L = im.shape[axis]
    A = np.prod(im.shape) / im.shape[axis]
    g = (results.effective_porosity * A) / (results.tortuosity * (L - 1))
    results.diffusive_conductance = g
    results.volume = np.prod(im.shape)
    results.axis = axis
    results.time = time.perf_counter() - t0
    return results


def tortuosity_map(im, block_size: int = None, axis: int = None, slices=None, dask_on=True):
    """
    Compute tortuosity and diffusive conductance on a set of blocks determined
    by the block size.

    Parameters
    ----------
    im : np.array
        The binary image to analyze with ``True`` indicating phase of interest.
    block_size : int
        The size of the blocks for the image to be subdivided into.
    slices : list
        A list containing slice objects for the image to be analyzed. If this is
        given the `block_size` is ignored.
    axis : int
        The axis along which to simulate the diffusion. If `None` then all
        axes are considered.

    Returns
    -------
    df_out : pd.DataFrame
        A dataframe containing information of all the blocks analyzed.

    Notes
    -----
    This is called by `rev_tortuosity` to queue up all the blocks to be analyzed.
    If both `block_size` and `slices` are left empty, the default mode of obtaining
    slices is set to `grid`.
    """
    if block_size is not None and slices is None:
        slices = get_slices_grid(im, block_size=block_size)

    if axis is None:
        axes = range(im.ndim)
    else:
        axes = [axis]
    results = []
    desc = inspect.currentframe().f_code.co_name  # Get current func name
    for s in tqdm(slices, desc=desc, **settings.tqdm):
        for ax in axes:
            if dask_on:
                tau_obj = dask.delayed(calc_g)(im[s], axis=ax)

            else:
                tau_obj = calc_g(im[s], axis=ax)

            if dask_on:
                tau_obj = tau_obj.compute()
            tau_obj.slice = s

            results.append(tau_obj)

    df_out = pd.DataFrame()
    df_out["eps_orig"] = [r.original_porosity for r in results]
    df_out["eps_perc"] = [r.effective_porosity for r in results]
    df_out["g"] = [r.diffusive_conductance for r in results]
    df_out["tau"] = [r.tortuosity for r in results]
    df_out["volume"] = [r.volume for r in results]
    df_out["length"] = [block_size for r in results]
    df_out["axis"] = [r.axis for r in results]
    df_out["time"] = [r.time for r in results]
    df_out["slice"] = [r.slice for r in results]

    return df_out


def rev_tortuosity(im, n=100, axis=None, slices=None, dask_on=False):
    """
    Calculates the tortuosity for a range of subdomain sizes suitable for an REV plot

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating the phase of interest
    n : int, default = 1000
        The number of random blocks to analyze. If `slices` are provided this is
        ignored.
    slices : list
        A list of `slice` objects into the image which define the subdomains. If
        not provided then `n` random blocks are used.
    axis : int
        The axis along which to simulate the diffusion. If `None` then all
        axes are considered.

    Returns
    -------
    df : DataFrame
        A `pandas` data frame with the properties for each block on a given row

        ========== ==================================================================
        Attribute  Description
        ========== ==================================================================
        eps_orig   The original porosity of the subdomain tested
        eps_perc   The porosity of the subdomain after filling non-percolating
                   paths (e.g. blind and closed pores). This is the domain on which
                   the calculation of tortuosity is performed.
        g          The calculated diffusive conductance for the subdomain tested
        tau        The calculated tortuosity for the tested subdomain
        volume     The total volume of each cubic subdomain tested
        length     The length of one side of the subdomain tested
        axis       The axis for which the above properties were calculated
        time       The elapsed time required to perform the calculations
        slice      The coordinates for the subdomain tested in the original image
        ========== ==================================================================

    Examples
    --------
    `Click here
    <https://porespy.org/examples/metrics/reference/rev_tortuosity.html>`_
    to view online example.

    """
    if slices is None:
        slices = get_slices_random(im=im, n=n)
    df = tortuosity_map(im, block_size=None, axis=axis, slices=slices, dask_on=dask_on)

    profile = Results()
    profile.porosity_orig = df["eps_orig"]
    profile.porosity_perc = df["eps_perc"]
    profile.g = df["g"]
    profile.tau = df["tau"]
    profile.volume = df["volume"]
    profile.length = df["length"]
    profile.axis = df["axis"]
    profile.time = df["time"]
    profile.slice = df["slice"]
    return profile


def rev_plot(df: pd.DataFrame, size: int, figsize: list = [10, 7]):
    """
    Creates REV plot from the output of `rev_tortuosity`.

    Parameters
    ----------
    df : pd.DataFrame
        The output of `rev_tortuosity`.
    size : int
        The length of one side of the cube image.
    fig_size : list
        The size of the figure to be outputted. Default to [10,7].

    Returns
    -------
    all_fig : list
        A list containing all of the matplotlib figure handles.
    all_ax : list
        A list containing all of the matplotlib axes handles.

    Notes
    -----
    All values of "np.inf" are treated as the next highest tortuosity within that bin
    purely for the sake of plotting. The original dataset is not altered.

    """

    import matplotlib.pyplot as plt

    from porespy.visualization import set_mpl_style

    set_mpl_style()

    all_fig = []
    all_ax = []

    for i, axis in enumerate(np.unique(df["axis"])):
        fig, axes = plt.subplots(figsize=figsize)

        # filter for one axis
        tmp = df[df["axis"] == axis]

        data = []
        vol_frac = []

        for vol in np.unique(tmp["volume"]):
            taus = tmp[tmp["volume"] == vol]["tau"]

            unique_tau = sorted(set(taus), reverse=True)

            if len(unique_tau) > 1:
                highest = unique_tau[1]

            else:
                highest = unique_tau[0] if unique_tau else 0

            taus = taus.replace([np.inf], highest)

            data.append(np.log10(taus))
            vol_frac.append(np.log10(vol / (size ** (len(np.unique(df["axis"]))))))

        axes.violinplot(data, vol_frac, widths=0.1)
        axes.set_title(f"REV: Axis {axis}")
        axes.set_xlabel("Normalized Volume Fraction")
        axes.set_ylabel(r"log$_{10}$($\tau$)")

        all_fig.append(fig)
        all_ax.append(axes)

    return all_fig, all_ax


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    import porespy as ps
    ps.settings.tqdm['disable'] = False
    ps.settings.tqdm['leave'] = True

    im = ps.generators.blobs([300] * 2, porosity=0.7, blobiness=2, seed=1)
    # im = ps.generators.random_cantor_dust([500, 500])

    # slices = ps.tools.get_slices_multigrid(im, [40, 300])
    slices = ps.tools.get_slices_random(im, 500)
    rev = ps.metrics.rev_tortuosity(im, slices=slices, axis=0, dask_on=True)
    converted = ps.tools.results_to_df(rev)
    poro = ps.metrics.rev_porosity(im, slices=slices)

    # %%
    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(
        poro.volume, poro.porosity, marker=".", alpha=0.25, fc="tab:red", ec="none"
    )
    ax[1].scatter(
        rev.volume[rev.axis == 0],
        rev.tau[rev.axis == 0],
        marker=".",
        alpha=0.25,
        fc="tab:blue",
        ec="none",
    )
    ax[2].scatter(
        rev.porosity_perc[rev.axis == 0],
        rev.tau[rev.axis == 0],
        marker=".",
        alpha=0.25,
        fc="tab:green",
        ec="none",
    )
    ax[0].set_ylim([0, 1])
    ax[0].set_xlim([0, im.size])
    ax[0].set_ylabel("Porosity")
    ax[0].set_xlabel("Subdomain Volume")
    ax[1].set_ylim([0, None])
    ax[1].set_xlim([0, im.size])
    ax[1].set_ylabel("log(Tortuosity)")
    ax[1].set_xlabel("Subdomain Volume")
    ax[2].set_xlim([0, 1])
    ax[2].set_ylim([0, None])
    ax[2].set_xlabel("Porosity")
    ax[2].set_ylabel("log(Tortuosity)")
