import numpy as np
import scipy.ndimage as spim
from GenericTest import GenericTest
from skimage.morphology import footprint_rectangle

import porespy as ps
from porespy.tools import get_edt

edt = get_edt()
ps.settings.tqdm['disable'] = True


class IBIPTest(GenericTest):

    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.lattice_spheres(shape=[101, 101],
                                                r=5, spacing=25,
                                                offset=[5, 5], lattice='tri')
        bd = np.zeros_like(self.im)
        bd[:, 0] = True
        self.bd = bd
        self.im2D = ps.generators.blobs(shape=[51, 51],
                                        seed=0,
                                        porosity=0.48212226066897346,
                                        periodic=False,
                                        )
        assert self.im2D.sum()/self.im2D.size == 0.48212226066897346
        self.im3D = ps.generators.blobs(shape=[51, 51, 51],
                                        seed=0,
                                        porosity=0.49954391599007925,
                                        periodic=False,
                                        )
        assert self.im3D.sum()/self.im3D.size == 0.49954391599007925

    def sc_lattice_with_trapped_region(self):
        im = np.copy(self.im)
        crds = ps.generators.line_segment([64, 64], [49, 80])
        im[tuple(crds)] = False
        crds = ps.generators.line_segment([30, 64], [49, 80])
        im[tuple(crds)] = False
        im = ~spim.binary_dilation(~im, structure=footprint_rectangle((3, 3)))
        return im

    def test_ibip_equals_qbip(self):
        x = ps.simulations.ibip(self.im, inlets=self.bd)
        temp1 = x.im_seq
        pc = ps.filters.capillary_transform(im=self.im)
        y = ps.simulations.qbip(self.im, inlets=self.bd, pc=pc, conn='min')
        temp2 = ps.tools.make_contiguous(y.im_seq)
        assert np.all(temp1 == temp2)

    def test_ibip(self):
        # The test value below was changed since ibip no longer
        # convert the dt to ints, which allows it to match qbip
        # perfectly
        x = ps.simulations.ibip(self.im, inlets=self.bd)
        assert x.im_seq.max() == 268  # 318

    def test_ibip_w_trapping(self):
        # The test value below was changed since ibip no longer
        # convert the dt to ints, which allows it to match qbip
        # perfectly
        im = self.sc_lattice_with_trapped_region()
        outlets = ps.generators.borders(shape=im.shape, mode='faces')
        x = ps.simulations.ibip(im, inlets=self.bd)
        assert x.im_seq.max() == 402  # 391

        # The following asserts have been updated to 840 because the
        # find_trapped_regions function no longer accepts bins, and instead uses
        # ALL the values to generate the bins.
        inv_w_trapping = ps.filters.find_trapped_clusters(
            im=im,
            outlets=outlets,
            seq=x.im_seq,
            method='queue',
        )
        assert inv_w_trapping.sum() == 840


if __name__ == '__main__':
    self = IBIPTest()
    self.run_all()
