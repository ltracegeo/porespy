import matplotlib.pyplot as plt
import numpy as np

import porespy as ps

edt = ps.tools.get_edt()


def test_imbibition(plot=False):
    im = ps.generators.blobs(
        shape=[500, 500],
        porosity=0.708328,
        blobiness=2.5,
        seed=7,
    )
    inlets = np.zeros_like(im)
    inlets[-1, :] = True
    outlets = np.zeros_like(im)
    outlets[0, :] = True
    im = ps.filters.trim_nonpercolating_paths(
        im=im,
        inlets=inlets,
        outlets=outlets,
    )

    voxel_size = 1e-4
    sigma = 0.072
    theta = 180
    delta_rho = 1000
    g = 0
    bg = 'w'

    pc = ps.filters.capillary_transform(
        im=im,
        sigma=sigma,
        theta=theta,
        g=g,
        rho_nwp=delta_rho,
        rho_wp=0,
        voxel_size=voxel_size,
    )

    # Test basic drainage
    imb1 = ps.simulations.imbibition(im=im,
                                     pc=pc,
                                     inlets=inlets,)
    # No trapping occurred
    assert imb1.im_pc[im].max() < np.inf
    assert imb1.im_snwp[im].max() < 1
    assert imb1.im_seq[im].min() > -1
    # No residual phase present
    assert imb1.im_pc[im].min() > -np.inf
    assert imb1.im_snwp[im].min() == 0.0
    assert imb1.im_seq[im].min() > 0

    # Test drainage with trapping
    imb2 = ps.simulations.imbibition(im=im,
                                     pc=pc,
                                     inlets=inlets,
                                     outlets=outlets,)
    # Some trapping occurred
    assert imb2.im_pc[im].min() == -np.inf
    assert imb2.im_snwp[im].min() == -1
    assert imb2.im_seq[im].min() == -1
    # No residual phase present
    assert imb2.im_pc[im].min() < np.inf
    assert imb2.im_snwp[im].max() < 1.0
    assert 0 not in imb2.im_seq[im]

    # Test drainage with residual, but no trapping
    drn = ps.simulations.drainage(im=im,
                                  pc=pc,
                                  inlets=outlets,
                                  outlets=inlets,)
    residual_nwp = drn.im_pc == np.inf
    swp_r = residual_nwp.sum()/im.sum()
    imb3 = ps.simulations.imbibition(im=im,
                                     pc=pc,
                                     inlets=inlets,
                                     residual=residual_nwp,)
    # No trapping occurred
    assert imb3.im_pc[im].min() > -np.inf
    assert imb3.im_snwp[im].min() == 0.0
    assert -1 not in imb3.im_seq[im]
    # Some residual phase present
    assert imb3.im_pc[im].max() == np.inf
    assert imb3.im_snwp[im].max() == (1 - swp_r)
    assert imb3.im_seq[im].min() == 0

    # Test drainage with residual and trapping
    imb4 = ps.simulations.imbibition(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
        residual=residual_nwp,
    )
    # Some trapping occurred
    assert imb4.im_pc[im].min() == -np.inf
    assert imb4.im_snwp[im].min() == -1  # Trapped
    assert imb4.im_seq[im].min() == -1
    # Some residual phase present
    assert imb4.im_pc[im].max() == np.inf
    assert imb4.im_snwp[im].max() < 1.0
    assert (imb4.im_seq[im] == 0).sum() > 0

    # Now confirm that pc_map_to_pc_curve is correct
    # No trapping or residual
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        im=im, pc=imb1.im_pc, seq=imb1.im_seq, mode='imbibition')
    assert Pc.min() > -np.inf
    assert Pc.max() < np.inf
    assert Snwp.min() == 0
    # assert Snwp.max() == 1.0

    # Trapping, no residual
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        im=im, pc=imb2.im_pc, seq=imb2.im_seq, mode='imbibition')
    assert Pc.max() < np.inf
    assert Pc.min() == -np.inf
    assert Snwp.min() > 0
    # assert Snwp.max() == 1.0

    # Residual, no trapping
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        im=im, pc=imb3.im_pc, seq=imb3.im_seq, mode='imbibition')
    assert Pc.min() > -np.inf
    assert Pc.max() == np.inf
    assert Snwp.max() == (1 - swp_r)
    assert Snwp.min() == 0.0

    # Residual and trapping
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        im=im, pc=imb4.im_pc, seq=imb4.im_seq, mode='imbibition')
    assert Pc.min() == -np.inf
    assert Pc.max() == np.inf
    assert Snwp.min() > 0
    assert Snwp.max() == (1 - swp_r)

    # %% Visualize the invasion configurations for each scenario
    if plot:
        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(imb1.im_snwp/im, origin='lower')
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(imb2.im_snwp/im, origin='lower')
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(imb3.im_snwp/im, origin='lower')
        ax[1][0].set_title("No trapping, with residual")
        ax[1][1].imshow(imb4.im_snwp/im, origin='lower')
        ax[1][1].set_title("With trapping, with residual")

    # %% Plot the capillary pressure curves for each scenario
    if plot:
        fig, ax = plt.subplots(figsize=[5, 5])
        fig.set_facecolor(bg)
        ax.set_facecolor(bg)

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            im=im,
            pc=imb1.im_pc,
            seq=imb1.im_seq,
            mode='imbibition',
            pc_min=0,
            pc_max=200,
        )
        plt.step(
            Pc,
            Snwp,
            'b-o',
            where='post',
            label="No trapping, no residual",
        )

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            im=im,
            pc=imb2.im_pc,
            seq=imb2.im_seq,
            mode='imbibition',
            pc_min=0,
            pc_max=200,
        )
        plt.step(
            Pc,
            Snwp,
            'r--o',
            where='post',
            label="With trapping, no residual",
        )

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            im=im,
            pc=imb3.im_pc,
            seq=imb3.im_seq,
            mode='imbibition',
            pc_min=0,
            pc_max=200,
        )
        plt.step(
            Pc,
            Snwp,
            'g--o',
            where='post',
            label="No trapping, with residual",
        )

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            im=im,
            pc=imb4.im_pc,
            seq=imb4.im_seq,
            mode='imbibition',
            pc_min=0,
            pc_max=200,
        )
        plt.step(
            Pc,
            Snwp,
            'm--o',
            where='post',
            label="With trapping, with residual",
        )
        ax.set_xlabel('Capillary Pressure')
        ax.set_ylabel('Non-Wetting Phase Saturation')
        plt.legend()


# %%
if __name__ == "__main__":
    test_imbibition(plot=False)
