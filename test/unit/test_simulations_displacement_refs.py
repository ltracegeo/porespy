import numpy as np

import porespy as ps
edt = ps.tools.get_edt()


class TestDisplacementRefs():

    def setup_class(self):
        self.im = ~ps.generators.random_spheres(
            [300, 300], r=10, clearance=10, seed=0)
        self.inlets = ps.generators.faces(self.im.shape, inlet=0)
        self.outlets = ps.generators.faces(self.im.shape, outlet=0)

        self.im3D = ~ps.generators.random_spheres(
            [50, 50, 50], r=0, clearance=0, seed=0)
        self.inlets3D = ps.generators.faces(self.im3D.shape, inlet=0)
        self.outlets3D = ps.generators.faces(self.im3D.shape, outlet=0)

    def test_drainage_2D_no_trapping_smooth(self):
        drn = {}
        smooth = True
        dt = edt(self.im).astype(int)
        drn['dt'] = ps.simulations.drainage_dt(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            dt=dt,
        )
        drn['fft'] = ps.simulations.drainage_conv(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            dt=dt,
        )
        drn['bf'] = ps.simulations.drainage_bf(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            dt=dt,
        )
        drn['dt_fft'] = ps.simulations.drainage_dt_conv(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            dt=dt,
        )

        a, b = 'bf', 'dt'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'bf', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'bf', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'fft', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

    def test_drainage_2D_no_trapping_not_smooth(self):
        drn = {}
        smooth = False
        # Steps must be integers for all methods to match
        steps = np.arange(1, 40, 1)
        drn['dt'] = ps.simulations.drainage_dt(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        drn['fft'] = ps.simulations.drainage_conv(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        drn['bf'] = ps.simulations.drainage_bf(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        drn['dt_fft'] = ps.simulations.drainage_dt_conv(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

    def test_drainage_3D_no_trapping_not_smooth(self):
        drn = {}
        smooth = False
        # Steps must be integers for all methods to match
        steps = np.arange(1, 40, 1)
        drn['dt'] = ps.simulations.drainage_dt(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['fft'] = ps.simulations.drainage_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['bf'] = ps.simulations.drainage_bf(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['dt_fft'] = ps.simulations.drainage_dt_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

    def test_drainage_3D_no_trapping_smooth(self):
        drn = {}
        smooth = True
        # Steps must be integers for all methods to match
        steps = np.arange(1, 40, 1)
        drn['dt'] = ps.simulations.drainage_dt(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['fft'] = ps.simulations.drainage_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['bf'] = ps.simulations.drainage_bf(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['dt_fft'] = ps.simulations.drainage_dt_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

    def test_drainage_2D_w_trapping_smooth(self):
        drn = {}
        smooth = True
        # Steps must be integers for all methods to match
        steps = np.arange(1, 40, 1)
        drn['dt'] = ps.simulations.drainage_dt(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        drn['fft'] = ps.simulations.drainage_conv(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        drn['bf'] = ps.simulations.drainage_bf(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        drn['dt_fft'] = ps.simulations.drainage_dt_conv(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

    def test_drainage_2D_w_trapping_not_smooth(self):
        drn = {}
        smooth = False
        # Steps must be integers for all methods to match
        steps = np.arange(1, 40, 1)
        drn['dt'] = ps.simulations.drainage_dt(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        drn['fft'] = ps.simulations.drainage_conv(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        drn['bf'] = ps.simulations.drainage_bf(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        drn['dt_fft'] = ps.simulations.drainage_dt_conv(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

    def test_drainage_3D_w_trapping_not_smooth(self):
        drn = {}
        smooth = False
        # Steps must be integers for all methods to match
        steps = np.arange(1, 40, 1)
        drn['dt'] = ps.simulations.drainage_dt(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['fft'] = ps.simulations.drainage_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['bf'] = ps.simulations.drainage_bf(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['dt_fft'] = ps.simulations.drainage_dt_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

    def test_drainage_3D_w_trapping_smooth(self):
        drn = {}
        smooth = True
        # Steps must be integers for all methods to match
        steps = np.arange(1, 40, 1)
        drn['dt'] = ps.simulations.drainage_dt(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['fft'] = ps.simulations.drainage_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['bf'] = ps.simulations.drainage_bf(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        drn['dt_fft'] = ps.simulations.drainage_dt_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = drn[a].im_seq == drn[b].im_seq
        assert np.all(tmp)
        tmp = drn[a].im_size == drn[b].im_size
        assert np.all(tmp)

    def test_imbibition_2D_no_trapping_smooth(self):
        imb = {}
        smooth = True
        steps = np.arange(39, 0, -1)
        imb['bf'] = ps.simulations.imbibition_bf(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        imb['dt'] = ps.simulations.imbibition_dt(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        imb['fft'] = ps.simulations.imbibition_conv(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        imb['dt_fft'] = ps.simulations.imbibition_dt_conv(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

    def test_imbibition_2D_no_trapping_not_smooth(self):
        imb = {}
        smooth = False
        steps = np.arange(39, 0, -1)
        imb['bf'] = ps.simulations.imbibition_bf(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        imb['dt'] = ps.simulations.imbibition_dt(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        imb['fft'] = ps.simulations.imbibition_conv(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )
        imb['dt_fft'] = ps.simulations.imbibition_dt_conv(
            im=self.im,
            inlets=self.inlets,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

    def test_imbibition_3D_no_trapping_smooth(self):
        imb = {}
        smooth = True
        steps = np.arange(39, 0, -1)
        imb['bf'] = ps.simulations.imbibition_bf(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['dt'] = ps.simulations.imbibition_dt(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['fft'] = ps.simulations.imbibition_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['dt_fft'] = ps.simulations.imbibition_dt_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

    def test_imbibition_3D_no_trapping_not_smooth(self):
        imb = {}
        smooth = False
        steps = np.arange(39, 0, -1)
        imb['bf'] = ps.simulations.imbibition_bf(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['dt'] = ps.simulations.imbibition_dt(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['fft'] = ps.simulations.imbibition_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['dt_fft'] = ps.simulations.imbibition_dt_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

    def test_imbibition_2D_w_trapping_smooth(self):
        imb = {}
        smooth = True
        steps = np.arange(39, 0, -1)
        imb['bf'] = ps.simulations.imbibition_bf(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        imb['dt'] = ps.simulations.imbibition_dt(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        imb['fft'] = ps.simulations.imbibition_conv(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        imb['dt_fft'] = ps.simulations.imbibition_dt_conv(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

    def test_imbibition_2D_w_trapping_not_smooth(self):
        imb = {}
        smooth = False
        steps = np.arange(39, 0, -1)
        imb['bf'] = ps.simulations.imbibition_bf(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        imb['dt'] = ps.simulations.imbibition_dt(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        imb['fft'] = ps.simulations.imbibition_conv(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )
        imb['dt_fft'] = ps.simulations.imbibition_dt_conv(
            im=self.im,
            inlets=self.inlets,
            outlets=self.outlets,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

    def test_imbibition_3D_w_trapping_smooth(self):
        imb = {}
        smooth = True
        steps = np.arange(39, 0, -1)
        imb['bf'] = ps.simulations.imbibition_bf(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['dt'] = ps.simulations.imbibition_dt(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['fft'] = ps.simulations.imbibition_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['dt_fft'] = ps.simulations.imbibition_dt_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

    def test_imbibition_3D_w_trapping_not_smooth(self):
        imb = {}
        smooth = False
        steps = np.arange(39, 0, -1)
        imb['bf'] = ps.simulations.imbibition_bf(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['dt'] = ps.simulations.imbibition_dt(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['fft'] = ps.simulations.imbibition_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )
        imb['dt_fft'] = ps.simulations.imbibition_dt_conv(
            im=self.im3D,
            inlets=self.inlets3D,
            outlets=self.outlets3D,
            smooth=smooth,
            steps=steps,
        )

        a, b = 'dt', 'bf'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'dt_fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)

        a, b = 'dt', 'fft'
        tmp = imb[a].im_seq == imb[b].im_seq
        assert np.all(tmp)
        tmp = imb[a].im_size == imb[b].im_size
        assert np.all(tmp)


if __name__ == '__main__':
    t = TestDisplacementRefs()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test_'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
