from porespy.filters import capillary_transform
from porespy.generators import borders
from porespy.metrics import pc_map_to_pc_curve
from porespy.simulations import drainage, imbibition
from porespy.tools import Results

__all__ = [
    'hg_porosimetry',
]


def hg_porosimetry(im, steps=25, voxel_size=1.0):
    r"""
    Simulates mercury intrusion and extrusion experiment

    Parameters
    ----------
    im : ndarray
        The boolean image of the void space
    steps : int
        The number of pressure steps to apply
    voxel_size : float
        The voxel size of the image in units of `m/vx side`. This is
        use to compute the capillary transform, with all other inputs
        being those for mercury.

    Returns
    -------
    results : dataclass
        An object with the following attributes:

        ---------------- ----------------------------------------------------
        Attribute        Description
        ---------------- ----------------------------------------------------
        pc_intrusion     Capillary pressure during the intrusion simulation
        snwp_intrusion   Non-wetting phase saturations during the intrusion
        pc_extrusion     Capillary pressure during the extrusion simulation
        snwp_extrusion   Non-wetting phase saturations during the extrusion
        ---------------- ----------------------------------------------------

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/hg_porosimetry.html>`_
    to view online example.

    """
    faces = borders(im.shape, mode='faces')
    pc = capillary_transform(
        im=im,
        sigma=0.465,
        theta=140,
        voxel_size=voxel_size,
    )

    drn = drainage(im=im, pc=pc, inlets=faces, steps=steps)
    imb = imbibition(im=im, pc=pc, outlets=faces, steps=steps)

    res = Results()
    pc, snwp = pc_map_to_pc_curve(
        im=im,
        pc=drn.im_pc,
        seq=drn.im_seq,
        mode='drainage',
        fix_ends=True,
    )
    res.pc_intrusion = pc
    res.snwp_intrusion = snwp

    pc, snwp = pc_map_to_pc_curve(
        im=im,
        pc=imb.im_pc,
        seq=imb.im_seq,
        mode='imbibition',
        fix_ends=True,
    )
    res.pc_extrusion = pc
    res.snwp_extrusion = snwp

    return res


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import porespy as ps

    i = 50591
    # i = np.random.randint(0, 100000, 1)
    print(i)
    voxel_size = 1e-5
    steps = 50
    im = ps.generators.blobs([100, 100, 100], porosity=0.6, seed=i)
    mip = hg_porosimetry(im, voxel_size=voxel_size, steps=steps)

    fig, ax = plt.subplots()
    ax.step(np.log10(mip.pc_intrusion), mip.snwp_intrusion, 'b.-', where='post')
    ax.step(np.log10(mip.pc_extrusion), mip.snwp_extrusion, 'r.-', where='post')
    ax.set_ylim([0, 1.05])

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(drn.im_seq/im)
    # ax[1].imshow(imb.im_seq/im)
