import pytest
import numpy as np
import scipy.ndimage as spim
import porespy as ps

ps.settings.tqdm['disable'] = True
try:
    from pyedt import edt
except ModuleNotFoundError:
    from edt import edt


class SnowsTest:
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100],
                                      porosity=0.5,
                                      blobiness=1)
        self.dt = edt(self.im)

    def test_trim_saddle_points_legacy(self):
        peaks = ps.filters.find_peaks(self.dt)
        peaks2 = ps.filters.trim_saddle_points_legacy(peaks, self.dt)
        assert peaks.sum() >= peaks2.sum()

    def test_watershed_stitching(self):
        from porespy.filters._snows import _watershed_stitching

        img = np.array([
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
            [0,  1,  2,  3,  4,  3,  4,  5,  6,  0],
            [0,  7,  8,  9, 10,  9, 10, 11, 12,  0],
            [0, 13, 14, 15, 16, 15, 16, 17, 18,  0],
            [0, 19, 20, 21, 22, 21, 22, 23, 24,  0],
            [0, 13, 14, 15, 16, 15, 16, 17, 18,  0],
            [0, 19, 20, 21, 22, 21, 22, 23, 24,  0],
            [0, 25, 26, 27, 28, 27, 28, 29, 30,  0],
            [0, 31, 32, 33, 34, 33, 34, 35, 36,  0],
            [0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
        ])

        expected_img = np.array([
            [ 1,  2,  3,  4,  5,  6],
            [ 7,  8,  9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36],
        ])

        chunk_shape = (3, 3)

        stitched_img = _watershed_stitching(im=img.copy(), chunk_shape=chunk_shape)

        assert np.all(stitched_img == expected_img)

    def test_trim_variable_overlap(self):
        from porespy.filters._snows import _trim_variable_overlap

        img = np.array([
            [ 1,  2,  3,  4,  5,  6],
            [ 7,  8,  9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36],
        ])

        overlaps = np.array([[0, 3], [0, 3]], dtype=int)

        expected_img = np.array([
            [ 1,  2,  3,  4],
            [ 7,  8,  9, 10],
            [13, 14, 15, 16],
            [19, 20, 21, 22],
        ])

        trimmed_img = _trim_variable_overlap(img, overlaps)

        assert np.all(trimmed_img == expected_img)

    def test_get_chunk_with_overlap(self):
        from porespy.filters._snows import _get_chunk_with_overlap

        img = np.array([
            [ 1,  2,  3,  4,  5,  6],
            [ 7,  8,  9, 10, 11, 12],
            [13, 14, 15, 16, 17, 18],
            [19, 20, 21, 22, 23, 24],
            [25, 26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35, 36],
        ])

        chunk_shape = (3, 3)

        chunk_index = (0, 0)
        max_radii_map = np.array([[2, 2], [2, 2]])
        chunk_with_overlap, overlaps = _get_chunk_with_overlap(img, chunk_index, chunk_shape, max_radii_map)

        expected_img = np.array([
            [ 1,  2,  3,  4,  5],
            [ 7,  8,  9, 10, 11],
            [13, 14, 15, 16, 17],
            [19, 20, 21, 22, 23],
            [25, 26, 27, 28, 29],
        ])
        expected_overlap = np.array([[0, 2], [0, 2]])

        assert np.all(chunk_with_overlap == expected_img)
        assert np.all(overlaps == expected_overlap)

    def test_trim_saddle_points_matches_dense_reference(self):
        for ndim, sigma, r_max in [(n, s, r) for n in (2, 3) for s, r in [(0, 2), (0.4, 4), (1.3, 6)]]:
            im, dt = _random_pores(ndim, seed=ndim)
            blurred = spim.gaussian_filter(dt, sigma) * im if sigma > 0 else dt
            peaks = ps.filters.find_peaks(blurred, r_max=r_max)
            for maxiter in (3, 20):
                expected = _trim_saddle_points_dense(peaks, dt, maxiter=maxiter)
                result = ps.filters.trim_saddle_points(peaks, dt, maxiter=maxiter)
                assert result.dtype == expected.dtype
                assert np.array_equal(result, expected)
            labels = spim.label(peaks)[0]
            assert np.array_equal(ps.filters.trim_saddle_points(labels, dt),
                                  _trim_saddle_points_dense(labels, dt))

    def test_trim_saddle_points_without_peaks_in_pores(self):
        dt = np.zeros((10, 10, 10))
        peaks = np.ones_like(dt, dtype=bool)  # e.g. find_peaks on an all-solid image
        assert not ps.filters.trim_saddle_points(peaks, dt).any()
        assert not ps.filters.trim_saddle_points(np.zeros_like(peaks), dt).any()

    def test_find_pore_peaks_matches_find_peaks(self):
        from porespy.filters._snows import _find_pore_peaks

        for ndim, r_max in [(n, r) for n in (2, 3) for r in (2, 4, 6)]:
            im, dt = _random_pores(ndim, seed=10 + ndim)
            expected = ps.filters.find_peaks(dt, r_max=r_max) & im
            assert np.array_equal(_find_pore_peaks(dt, r_max), expected)


def _random_pores(ndim, seed):
    rng = np.random.default_rng(seed)
    shape = (200, 200) if ndim == 2 else (60, 70, 50)
    im = spim.gaussian_filter(rng.random(shape), 2.0)
    im = im > np.quantile(im, 0.85)  # many unconnected pores
    return im, edt(im)


def _trim_saddle_points_dense(peaks, dt, maxiter=20):
    # Original per-cluster implementation of trim_saddle_points, kept as reference
    from skimage.morphology import footprint_rectangle
    from porespy.tools import extend_slice

    new_peaks = np.zeros_like(peaks, dtype=bool)
    strel = footprint_rectangle((3,) * dt.ndim)
    labels, _ = spim.label(peaks > 0)
    for i, s in enumerate(spim.find_objects(labels)):
        sx = extend_slice(s, shape=peaks.shape, pad=maxiter)
        peaks_i = labels[sx] == i + 1
        dt_i = dt[sx]
        im_i = dt_i > 0
        for _ in range(maxiter):
            peaks_dil = spim.binary_dilation(input=peaks_i, structure=strel)
            peaks_max = peaks_dil * np.amax(dt_i * peaks_dil)
            peaks_extended = (peaks_max == dt_i) * im_i
            if np.all(peaks_extended == peaks_i):
                new_peaks[sx] += peaks_i
                break
            elif np.sum(peaks_extended * peaks_i, dtype=np.int64) == 0:
                break
            peaks_i = peaks_extended
    return new_peaks * peaks


if __name__ == '__main__':
    t = SnowsTest()
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
