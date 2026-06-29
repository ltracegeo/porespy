import numpy as np

import porespy as ps

edt = ps.tools.get_edt()
ps.settings.loglevel = "CRITICAL"
ps.settings.tqdm['disable'] = True


class ParallelTest():
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100],
                                      porosity=0.499829,
                                      blobiness=2,
                                      periodic=False,)
        # Ensure that im was generated as expeccted
        assert ps.metrics.porosity(self.im) == 0.499829
        self.im_dt = edt(self.im)

    def test_find_peaks_2D(self):
        im = ps.generators.blobs(shape=[200, 200], blobiness=2, periodic=False,)
        dt = edt(im)
        mx_serial = ps.filters.find_peaks(dt=dt)
        parallel_kw = {"divs": 2}
        mx_parallel_1 = ps.filters.find_peaks(dt=dt, parallel_kw=parallel_kw)
        assert np.all(mx_serial == mx_parallel_1)

    def test_find_peaks_3D(self):
        im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2, periodic=False,)
        dt = edt(im)
        mx_serial = ps.filters.find_peaks(dt=dt)
        parallel_kw = {"divs": 2}
        mx_parallel_1 = ps.filters.find_peaks(dt=dt, parallel_kw=parallel_kw)
        assert np.all(mx_serial == mx_parallel_1)

    def test_blobs_3D(self):
        np.random.seed(0)
        parallel_kw = {"divs": 1}
        im1 = ps.generators.blobs(shape=[101, 101, 101], parallel_kw=parallel_kw, periodic=False,)
        np.random.seed(0)
        parallel_kw = {"divs": 2}
        im2 = ps.generators.blobs(shape=[101, 101, 101], parallel_kw=parallel_kw, periodic=False,)
        assert np.all(im1 == im2)

    def test_blobs_2D(self):
        np.random.seed(0)
        s = 100
        parallel_kw = {"divs": 1}
        im1 = ps.generators.blobs(shape=[s, s], parallel_kw=parallel_kw, porosity=.5, periodic=False,)
        np.random.seed(0)
        parallel_kw = {"divs": 2}
        im2 = ps.generators.blobs(shape=[s, s], parallel_kw=parallel_kw, porosity=.5, periodic=False,)
        assert np.sum(im1 != im2) < 5


if __name__ == '__main__':
    t = ParallelTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print('running test: '+item)
            t.__getattribute__(item)()
