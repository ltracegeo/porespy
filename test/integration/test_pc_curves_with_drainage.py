import matplotlib.pyplot as plt
import numpy as np

import porespy as ps

edt = ps.tools.get_edt()


def test_drainage(plot=False):
    im = ps.generators.blobs(
        shape=[500, 500],
        porosity=0.708328,
        blobiness=1.5,
        seed=6,
    )
    inlets = np.zeros_like(im)
    inlets[0, :] = True
    outlets = np.zeros_like(im)
    outlets[-1, :] = True
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
    drn1 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
    )
    # No trapping occurred
    assert drn1.im_pc[im].max() < np.inf
    assert drn1.im_snwp[im].max() == 1
    assert drn1.im_seq[im].min() > 0
    # No residual phase present
    assert drn1.im_pc[im].min() > -np.inf
    assert drn1.im_snwp[im].min() > 0
    assert drn1.im_seq[im].min() > 0

    # Test drainage with trapping
    drn2 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        outlets=outlets,
    )
    # Some trapping occurred
    assert drn2.im_pc[im].max() == np.inf
    assert drn2.im_snwp[im].max() < 1
    assert drn2.im_seq[im].min() == -1
    # No residual phase present
    assert drn2.im_pc[im].min() > -np.inf
    assert drn2.im_snwp[im].min() == -1
    assert drn2.im_seq[im].min() == -1

    # Test drainage with residual, but no trapping
    residual_nwp = ps.filters.local_thickness(im) > 25
    snwp_r = residual_nwp.sum()/im.sum()
    drn3 = ps.simulations.drainage(
        im=im,
        pc=pc,
        inlets=inlets,
        residual=residual_nwp,
    )
    # No trapping occurred
    assert drn3.im_pc[im].max() < np.inf
    assert drn3.im_snwp[im].max() == 1
    assert drn3.im_seq[im].min() > -1
    # Some residual phase present
    assert drn3.im_pc[im].min() == -np.inf
    assert drn3.im_snwp[im].min() == snwp_r
    assert drn3.im_seq[im].min() == 0

    # # Test drainage with residual and trapping
    # drn4 = ps.simulations.drainage(
    #     im=im,
    #     pc=pc,
    #     inlets=inlets,
    #     outlets=outlets,
    #     residual=residual_nwp,
    # )
    # # Some trapping occurred
    # assert drn4.im_pc[im].max() == np.inf
    # assert drn4.im_snwp[im].min() == -1  # Trapped
    # assert drn4.im_seq[im].min() == -1
    # # Some residual phase present
    # assert drn4.im_pc[im].min() == -np.inf
    # assert np.unique(drn4.im_snwp[im])[1] == snwp_r
    # assert (drn4.im_seq[im] == 0).sum() > 0

    # Now let's confirm that pc_map_to_pc_curve is correct
    # No trapping or residual
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn1.im_pc, seq=drn1.im_seq)
    assert Pc.min() > -np.inf
    assert Pc.max() < np.inf
    assert Snwp.min() == 0  # Added by function to plot looks correct
    assert Snwp.max() == 1.0

    # Trapping, no residual
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn2.im_pc, seq=drn2.im_seq)
    assert Pc.min() > -np.inf
    assert Pc.max() == np.inf
    assert Snwp.min() == 0  # Added by function to plot looks correct
    assert Snwp.max() < 1.0

    # Residual, no trapping
    Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn3.im_pc, seq=drn3.im_seq)
    assert Pc.min() == -np.inf
    assert Pc.max() < np.inf
    assert Snwp.min() == snwp_r
    assert Snwp.max() == 1.0

    # # Residual and trapping
    # Pc, Snwp = ps.metrics.pc_map_to_pc_curve(im=im, pc=drn4.im_pc, seq=drn4.im_seq)
    # assert Pc.min() == -np.inf
    # assert Pc.max() == np.inf
    # assert Snwp.min() == snwp_r
    # assert Snwp.max() < 1.0

    # %% Visualize the invasion configurations for each scenario
    if plot:
        fig, ax = plt.subplots(2, 2, facecolor=bg)
        ax[0][0].imshow(drn1.im_snwp/im, origin='lower')
        ax[0][0].set_title("No trapping, no residual")
        ax[0][1].imshow(drn2.im_snwp/im, origin='lower')
        ax[0][1].set_title("With trapping, no residual")
        ax[1][0].imshow(drn3.im_snwp/im, origin='lower')
        ax[1][0].set_title("No trapping, with residual")
        # ax[1][1].imshow(drn4.im_snwp/im, origin='lower')
        # ax[1][1].set_title("With trapping, with residual")

    # %% Plot the capillary pressure curves for each scenario
    if plot:
        fig, ax = plt.subplots(figsize=[5, 5])
        fig.set_facecolor(bg)
        ax.set_facecolor(bg)

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            im=im,
            pc=drn1.im_pc,
            seq=drn1.im_seq,
            pc_min=-1000,
            pc_max=1000,
        )
        plt.step(
            Pc,
            Snwp,
            'b-s',
            where='post',
            label="No trapping, no residual",
        )

        Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
            im=im,
            pc=drn2.im_pc,
            seq=drn2.im_seq,
            pc_min=-1000,
            pc_max=1000,
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
            pc=drn3.im_pc,
            seq=drn3.im_seq,
            pc_min=-100,
            pc_max=1000,
        )
        plt.step(
            Pc,
            Snwp,
            'g--o',
            where='post',
            label="No trapping, with residual",
        )

        # Pc, Snwp = ps.metrics.pc_map_to_pc_curve(
        #     im=im,
        #     pc=drn4.im_pc,
        #     seq=drn4.im_seq,
        #     pc_min=-200,
        #     pc_max=1000,
        # )
        # plt.step(
        #     Pc,
        #     Snwp,
        #     'm--^',
        #     where='post',
        #     label="With trapping, with residual",
        # )
        # ax.set_xlabel('Capillary Pressure')
        # ax.set_ylabel('Non-Wetting Phase Saturation')
        # plt.legend()


# %%
if __name__ == "__main__":
    test_drainage(plot=False)
