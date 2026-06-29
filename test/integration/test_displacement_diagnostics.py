import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as spim


def test_displacement_diagnostics(plot=False):

    ps.visualization.set_mpl_style()

    edt = ps.tools.get_edt()
    im = ps.generators.blobs(
        shape=[100, 100],
        porosity=0.7,
        blobiness=1,
        seed=16,
    )
    im = ps.filters.fill_invalid_pores(im)

    dt = edt(im)
    pc = 2/dt
    pc[~im] = 0
    steps = np.unique(dt.astype(int)[im])

    r = 5
    smooth_ero = True
    smooth_dil = smooth_ero

    # Actual erosion
    ero1 = spim.binary_erosion(
        im, structure=ps.tools.ps_round(r, im.ndim, smooth_ero), border_value=1)

    ero2 = ~spim.binary_dilation(
        ~im, structure=ps.tools.ps_round(r, im.ndim, smooth_ero))

    # Using dt to perform erosion
    ero3 = (dt >= r) if smooth_ero else (dt > r)
    ero3 *= im

    # Using erosion with DT then finding edges
    ero4 = (dt >= r) if smooth_ero else (dt > r)
    ero4 *= im
    edges = ero4 * (dt <= (r + 1)) * im

    assert np.all(ero1 == ero2)
    assert np.all(ero1 == ero3)
    assert np.all(ero1 == ero4)
    assert np.all(ero2 == ero3)
    assert np.all(ero2 == ero4)
    assert np.all(ero3 == ero4)

    # Actual dilation
    dil1 = spim.binary_dilation(
        ero1, structure=ps.tools.ps_round(r, im.ndim, smooth_dil))

    # Doing dilation by erosion of the inverted erosion
    dil2 = ~spim.binary_erosion(
        ~ero2, structure=ps.tools.ps_round(r, im.ndim, smooth_dil), border_value=1)

    # Using dt to perform erosion
    tmp = edt(~ero3)
    dil3 = ~(tmp >= r) if smooth_dil else ~(tmp > r)
    dil3 *= im

    # Adding spheres to edge sites, i.e. brute-force method
    coords = np.vstack(np.where(edges))
    dil4 = np.zeros_like(im)
    dil4 = ps.tools._insert_disk_at_points(
        dil4, coords=coords, r=r, v=True, smooth=smooth_dil)
    dil4 += ero4  # Adding erosion to dilation to ensure centers are filled

    assert np.all(dil1 == dil2)
    assert np.all(dil1 == dil3)
    assert np.all(dil1 == dil4)
    assert np.all(dil2 == dil3)
    assert np.all(dil2 == dil4)
    assert np.all(dil3 == dil4)

    if plot:
        fig, ax = plt.subplots(2, 4)
        ax[0][0].imshow(ero1 + ~im*0.4)
        ax[0][0].set_title('Erosion of Voids')
        ax[0][1].imshow(ero2 + ~im*0.4)
        ax[0][1].set_title('Erosion via Dilation of Solid')
        ax[0][2].imshow(ero3 + ~im*0.4)
        ax[0][2].set_title('DT Thresholding')
        ax[0][3].imshow((ero4 + edges*1.0) + ~im*0.4)
        ax[0][3].set_title('DT Thresholding + Edge Finding')
        ax[1][0].imshow((dil1 + ero1*1.0) + ~im*0.4)
        ax[1][0].set_title('Dilation of Erosion')
        ax[1][1].imshow((dil2 + ero2*1.0) + ~im*0.4)
        ax[1][1].set_title('Dilation via Erosion of Inverted Erosion')
        ax[1][2].imshow((dil3 + ero3*1.0) + ~im*0.4)
        ax[1][2].set_title('DT Thresholding')
        ax[1][3].imshow((dil4 + ero4*1.0) + ~im*0.4)
        ax[1][3].set_title('BF Dilation')


if __name__ == "__main__":
    test_displacement_diagnostics(plot=False)
