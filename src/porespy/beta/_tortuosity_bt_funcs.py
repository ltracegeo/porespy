import logging
import time

import dask
import numpy as np
import pandas as pd

import porespy as ps
from porespy.tools import Results, get_edt, get_tqdm

__all__ = [
    'tortuosity_bt',
    'get_block_sizes',
    'df_to_tortuosity',
    'rev_tortuosity',
]


logger = logging.getLogger()
tqdm = get_tqdm()
edt = get_edt()


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
        Dicionary of keyword arguments to pass on to the solver.  The most
        relevant one being `'tol'` which is 1e-6 by default. Using larger values
        might improve speed at the cost of accuracy.

    Returns
    -------
    results : dataclass-like
        An object with the results of the calculation as attributes.

    Notes
    -----
    This is intended to receive blocks of a larger image and is used by
    `tortuosity_bt`.
    """
    import openpnm as op

    from porespy.simulations import tortuosity_fd
    solver_args = {'tol': 1e-6} | solver_args
    solver = solver_args.pop('solver', None)
    t0 = time.perf_counter()

    try:
        solver = op.solvers.PyamgRugeStubenSolver(**solver_args)
        results = tortuosity_fd(im=im, axis=axis, solver=solver)
    except Exception:
        results = Results()
        results.effective_porosity = 0.0
        results.original_porosity = im.sum()/im.size
        results.tortuosity = np.inf
        results.time = time.perf_counter() - t0
    L = im.shape[axis]
    A = np.prod(im.shape)/im.shape[axis]
    g = (results.effective_porosity * A) / (results.tortuosity * (L - 1))
    results.diffusive_conductance = g
    results.volume = np.prod(im.shape)
    results.axis = axis
    results.time = time.perf_counter() - t0
    return results


def get_block_sizes(im, block_size_range=[10, 100]):
    """
    Finds all viable block sizes between lower and upper limits

    Parameters
    ----------
    im : np.array
        The binary image to analyze with ``True`` indicating phase of interest.
    block_size_range : sequence of 2 ints
        The [lower, upper] range of the desired block sizes. Default is [10, 100]

    Returns
    -------
    sizes : ndarray
        All the viable block sizes in the specified range

    Notes
    -----
    This is called by `rev_tortuosity` to determine what size blocks to use.
    """
    shape = im.shape
    Lmin, Lmax = block_size_range
    a = np.ceil(min(shape)/Lmax).astype(int)
    block_sizes = min(shape) // np.arange(a, 9999)  # Generate WAY more than needed
    block_sizes = np.unique(block_sizes[block_sizes >= Lmin])
    return block_sizes


def tortuosity_map(im, block_size: int, dask_on=True):
    """
    Compute tortuosity and diffusive conductance on a series
    of blocks determined by the block size.

    Parameters
    ----------
    im : np.array
        The binary image to analyze with ``True`` indicating phase of interest.
    block_size : int
        The size of the blocks for the image to be subdivided into.

    Returns
    -------
    df_out : pd.DataFrame
        A dataframe containing information of all the blocks analyzed.

    Notes
    -----
    This is called by `rev_tortuosity` to queue up all the blocks to be analyzed.
    """
    slices = ps.tools.get_slices_grid(im, block_size=block_size)
    tmp = np.zeros(im.shape)

    results = []
    for s in tqdm(slices):
        for axis in range(im.ndim):
            if dask_on:
                tau_obj = dask.delayed(calc_g)(im[s], axis=axis)

            else:
                tau_obj = calc_g(im[s], axis=axis)

            tau_obj = tau_obj.compute()
            tau_obj.slice = s

            results.append(tau_obj)

    df_out = pd.DataFrame()

    df_out['eps_orig'] = [r.original_porosity for r in results]
    df_out['eps_perc'] = [r.effective_porosity for r in results]
    df_out['g'] = [r.diffusive_conductance for r in results]
    df_out['tau'] = [r.tortuosity for r in results]
    df_out['volume'] = [r.volume for r in results]
    df_out['length'] = [block_size for r in results]
    df_out['axis'] = [r.axis for r in results]
    df_out['time'] = [r.time for r in results]
    df_out['slice'] = [r.slice for r in results]

    return df_out

def rev_tortuosity(im, block_sizes=None, use_dask=True):
    """
    Generates the data for creating an REV plot based on tortuosity.

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest
    block_sizes : np.ndarray
        An array containing integers of block sizes to be calculated
    use_dask : bool
        A boolean determining the usage of `dask`.

    Returns
    -------
    df : DataFrame
        A `pandas` data frame with the properties for each block on a given row

        ========== ==================================================================
        Attribute  Description
        ========== ==================================================================
        eps_orig   The porosity of the subdomain tested
        eps_perc   The porosity of the subdomain tested after filling non-percolating paths
        g          The calculated diffusive conductance for the subdomain tested
        tau        The calculated tortuosity for the tested subdomain
        volume     The total volume of each cubic subdomain tested
        length     The length of one side of the subdomain tested
        axis       The axis for which the above properties were calculated
        time       The elapsed time required to perform the calculations
        slice      The coordinates for the subdomain tested in the original image
        ========== ==================================================================
    """
    all_dfs = []
    size = im.shape

    if block_sizes == None:
        block_sizes = get_block_sizes(im, [20, size[0]])

    for block in block_sizes:
        tmp = tortuosity_map(im, block, True)
        all_dfs.append(tmp)

    df = pd.concat(all_dfs)
    return df


def block_size_to_divs(shape, block_size):
    r"""
    Finds the number of blocks in each direction given the size of the blocks

    Parameters
    ----------
    shape : sequence of ints
        The [x, y, z] shape of the image
    block_size : int or sequence of ints
        The size of the blocks

    Returns
    -------
    divs : list of ints
        The number of blocks to divide the image into along each axis. The minimum
        number of blocks is 2.
    """
    shape = np.array(shape)
    divs = shape // np.array(block_size)
    # scraps = shape % np.array(block_size)
    divs = np.clip(divs, a_min=2, a_max=shape)
    return divs

def rev_plot(df: pd.DataFrame, size: int, figsize: list = [10, 7]):
    '''
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
    All values of "np.inf" are treated as the next highest tortuosity within that bin.

    '''

    import matplotlib.pyplot as plt
    ps.visualization.set_mpl_style()

    all_fig = []
    all_ax = []

    for i, axis in enumerate(np.unique(df['axis'])):
        fig, axes = plt.subplots(figsize=figsize)

        # filter for one axis
        tmp = df[df['axis'] == axis]

        data = []
        vol_frac = []

        for vol in np.unique(tmp['volume']):
            taus = tmp[tmp['volume'] == vol]["tau"]

            unique_tau = sorted(set(taus), reverse=True)

            if len(unique_tau) > 1:
                highest = unique_tau[1]

            else:
                highest = unique_tau[0] if unique_tau else 0

            taus = taus.replace([np.inf], highest)

            # if np.inf in taus and len(np.unique(taus) > 1):
            #     taus[taus==np.inf] = max(taus[taus!=np.inf])

            data.append(np.log10(taus))
            vol_frac.append(np.log10(vol / (size**(len(np.unique(df['axis']))))))

        axes.violinplot(data, vol_frac, widths=0.1)
        axes.set_title(f"REV: Axis {axis}")
        axes.set_xlabel("Normalized Volume Fraction")
        axes.set_ylabel(r"log$_{10}$($\tau$)")

        all_fig.append(fig)
        all_ax.append(axes)

    return all_fig, all_ax

def df_to_tortuosity(im, df):
    """
    Compute the tortuosity of a network populated with diffusive conductance values
    from the given dataframe.

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    df : dataframe
        The dataframe returned by the `blocks_to_dataframe` function
    block_size : int
        The size of the blocks used to compute the conductance values in `df`

    Returns
    -------
    tau : list of floats
        The tortuosity in all three principal directions
    """
    import openpnm as op
    block_size = list(df['length'])[0]
    divs = block_size_to_divs(shape=im.shape, block_size=block_size)

    net = op.network.Cubic(shape=divs)
    air = op.phase.Phase(network=net)
    gx = df['g'][df['axis'] == 0]
    gy = df['g'][df['axis'] == 1]
    gz = df['g'][df['axis'] == 2]

    g = np.hstack([gz, gy, gx])

    air['throat.diffusive_conductance'] = g

    bcs = {0: {'in': 'left', 'out': 'right'},
           1: {'in': 'front', 'out': 'back'},
           2: {'in': 'top', 'out': 'bottom'}}

    e = np.sum(im, dtype=np.int64) / im.size
    D_AB = 1
    tau = []

    for ax in range(im.ndim):
        fick = op.algorithms.FickianDiffusion(network=net, phase=air)
        fick.set_value_BC(pores=net.pores(bcs[ax]['in']), values=1.0)
        fick.set_value_BC(pores=net.pores(bcs[ax]['out']), values=0.0)
        fick.run()
        rate_inlet = fick.rate(pores=net.pores(bcs[ax]['in']))[0]
        L = (divs[ax] - 1) * block_size
        A = (np.prod(divs) / divs[ax]) * (block_size**2)
        D_eff = rate_inlet * L / (A * (1 - 0))
        tau.append(e * D_AB / D_eff)

    ws = op.Workspace()
    ws.clear()
    return tau


def tortuosity_bt(im, block_size=None, method="chords", use_dask=True):
    r"""
    Computes the tortuosity of an image in all directions

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int
        The size of the blocks which the image will be split into. If not provided,
        it will be determined by the provided method in `method`
    method : str
        The method to use to determine block sizes if `block_size` is not provided

        =========== ==================================================================
        method      description
        =========== ==================================================================
        'chords'    Uses `apply_chords_3D` from Porespy to determine the longest chord
                    possible in the image as the length of each block.
        'dt'        Uses the maximum length of the distance transform to determine
                    the length of each block.
        =========== ==================================================================

    use_dask : bool
        A boolean determining the usage of `dask` for parallel processing.
    """
    df = tortuosity_map(im, block_size, use_dask)
    tau = df_to_tortuosity(im, df)
    return tau


if __name__ == "__main__":
    import numpy as np

    import porespy as ps

    np.random.seed(1)

    im = ps.generators.blobs([100]*2)
    df = rev_tortuosity(im,)
    plots = rev_plot(df, 100)
