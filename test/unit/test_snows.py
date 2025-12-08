import pytest
import numpy as np
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


if __name__ == '__main__':
    t = SnowsTest()
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
