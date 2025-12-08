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

    # def test_snow_partitioning(self):
    #     snow = ps.filters.snow_partitioning(im=self.im, dt=self.dt)
    #     assert snow.regions.max() > 1
    #     # More specific assertions can be added here

    # def test_snow_partitioning_n(self):
    #     im = self.im.astype(int) + 1
    #     snow = ps.filters.snow_partitioning_n(im)
    #     assert snow.regions.max() > 1
    #     # More specific assertions can be added here

    # def test_snow_partitioning_parallel(self):
    #     snow = ps.filters.snow_partitioning_parallel(self.im, dt=self.dt)
    #     assert snow.regions.max() > 1
    #     # More specific assertions can be added here

    # def test_find_peaks(self):
    #     peaks = ps.filters.find_peaks(self.dt)
    #     assert peaks.sum() > 0
    #     # More specific assertions can be added here

    # def test_reduce_peaks(self):
    #     peaks = ps.filters.find_peaks(self.dt)
    #     peaks2 = ps.filters.reduce_peaks(peaks)
    #     assert peaks.sum() >= peaks2.sum()
    #     # More specific assertions can be added here

    # def test_trim_nearby_peaks(self):
    #     peaks = ps.filters.find_peaks(self.dt)
    #     peaks2 = ps.filters.trim_nearby_peaks(peaks, self.dt)
    #     assert peaks.sum() >= peaks2.sum()
    #     # More specific assertions can be added here

    # def test_trim_saddle_points(self):
    #     peaks = ps.filters.find_peaks(self.dt)
    #     peaks2 = ps.filters.trim_saddle_points(peaks, self.dt)
    #     assert peaks.sum() >= peaks2.sum()
    #     # More specific assertions can be added here

    def test_trim_saddle_points_legacy(self):
        peaks = ps.filters.find_peaks(self.dt)
        peaks2 = ps.filters.trim_saddle_points_legacy(peaks, self.dt)
        assert peaks.sum() >= peaks2.sum()
        # More specific assertions can be added here

    def test_snow_partitioning_parallel_variable(self):
        snow = ps.filters.snow_partitioning_parallel_variable(self.im, dt=self.dt, divs=2)
        assert isinstance(snow, ps.tools.Results)
        assert snow.regions.shape == self.im.shape
        assert snow.regions.max() > 1
        assert np.array_equal(snow.dt, self.dt)

    def test_get_chunk_with_variable_overlap(self):
        from porespy.filters._snows import _get_chunk_with_variable_overlap

        dt_large = np.arange(10*10*10).reshape(10, 10, 10)
        chunk_shape_large = (5, 5, 5)
        
        # Test case 1: first chunk (0,0,0) with uniform overlap of 1
        overlap_val = 1
        overlaps_full = np.array([[overlap_val, overlap_val]] * dt_large.ndim)
        chunk_idx_000 = (0, 0, 0)
        result_000 = _get_chunk_with_variable_overlap(dt_large, chunk_idx_000, chunk_shape_large, overlaps_full)
        
        expected_shape_000 = (chunk_shape_large[0] + overlap_val,
                              chunk_shape_large[1] + overlap_val,
                              chunk_shape_large[2] + overlap_val)
        assert result_000.shape == expected_shape_000
        np.testing.assert_array_equal(result_000, dt_large[0:6, 0:6, 0:6])

        # Test case 2: last chunk (1,1,1) with uniform overlap of 1
        chunk_idx_111 = (1, 1, 1)
        result_111 = _get_chunk_with_variable_overlap(dt_large, chunk_idx_111, chunk_shape_large, overlaps_full)
        
        expected_shape_111 = (chunk_shape_large[0] + overlap_val,
                              chunk_shape_large[1] + overlap_val,
                              chunk_shape_large[2] + overlap_val)
        assert result_111.shape == expected_shape_111
        np.testing.assert_array_equal(result_111, dt_large[4:10, 4:10, 4:10])

        # Test case 3: a middle chunk with uniform overlap
        dt_medium = np.arange(6*6*6).reshape(6, 6, 6)
        chunk_shape_medium = (2, 2, 2)
        overlap_val_medium = 1
        overlaps_medium_full = np.array([[overlap_val_medium, overlap_val_medium]] * dt_medium.ndim)
        chunk_idx_010 = (0, 1, 0)
        result_010 = _get_chunk_with_variable_overlap(dt_medium, chunk_idx_010, chunk_shape_medium, overlaps_medium_full)
        
        assert result_010.shape == (3, 4, 3) 
        np.testing.assert_array_equal(result_010, dt_medium[0:3, 1:5, 0:3])

        # Test case 4: first chunk with variable overlaps
        chunk_idx_var = (0, 0, 0)
        overlaps_var = np.array([[0, 2], [0, 1], [1, 1]]) # overlaps = [[left0, right0], [left1, right1], [left2, right2]]
        result_var = _get_chunk_with_variable_overlap(dt_large, chunk_idx_var, chunk_shape_large, overlaps_var)
        
        assert result_var.shape == (7, 6, 6)
        np.testing.assert_array_equal(result_var, dt_large[0:7, 0:6, 0:6])

    def test_trim_variable_overlap(self):
        from porespy.filters._snows import _trim_variable_overlap
        # Create a sample chunk with overlap
        chunk = np.ones((7, 6, 5))
        overlaps = np.array([[1, 2], [1, 1], [0, 1]])

        trimmed = _trim_variable_overlap(chunk, overlaps)

        # Expected shape after trimming:
        # axis 0: 7 - 1 - 2 = 4
        # axis 1: 6 - 1 - 1 = 4
        # axis 2: 5 - 0 - 1 = 4
        assert trimmed.shape == (4, 4, 4)

        # Test with a real data array to check content
        original_data = np.arange(4*4*4).reshape(4, 4, 4)
        # Pad it to create a chunk with overlaps
        chunk_with_padding = np.pad(original_data, ((1, 2), (1, 1), (0, 1)), mode='constant', constant_values=-1)
        assert chunk_with_padding.shape == (7, 6, 5)

        trimmed_again = _trim_variable_overlap(chunk_with_padding, overlaps)
        assert trimmed_again.shape == (4, 4, 4)
        np.testing.assert_array_equal(trimmed_again, original_data)

        # Another case in 2D
        chunk_2d = np.ones((10, 10))
        overlaps_2d = np.array([[2, 2], [3, 0]])
        trimmed_2d = _trim_variable_overlap(chunk_2d, overlaps_2d)
        assert trimmed_2d.shape == (6, 7)

    def test_stitch_chunks_variable(self):
        from porespy.filters._snows import _stitch_chunks_variable
        import scipy.ndimage as spim

        # Test case 1: Simple 2D image with 4 chunks, all labels should merge
        regions_2d_1 = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4]
        ], dtype=np.int32)
        chunk_shape_2d = (2, 2)
        divs_2d = (2, 2)
        
        stitched_regions_2d_1 = _stitch_chunks_variable(regions_2d_1, chunk_shape_2d, divs_2d)
        
        # All non-zero labels should merge into a single label (e.g., 1)
        expected_2d_1 = np.array([
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ], dtype=np.int32)
        # The labels will be resequenced, so we only compare the connectivity, not exact label values
        assert np.unique(stitched_regions_2d_1[regions_2d_1 > 0]).size == 1
        assert np.all(stitched_regions_2d_1[regions_2d_1 > 0] == stitched_regions_2d_1[1, 1])

        # Test case 2: 2D image with some labels merging, some not
        regions_2d_2 = np.array([
            [1, 1, 2, 0],
            [1, 1, 2, 0],
            [3, 3, 0, 4],
            [3, 3, 0, 4]
        ], dtype=np.int32)
        # Labels 1 and 2 should merge (boundary in X direction). Labels 1 and 3 should merge (boundary in Y direction).
        # Label 4 is isolated. So, 1, 2, 3 become one label, 4 stays separate.
        
        stitched_regions_2d_2 = _stitch_chunks_variable(regions_2d_2, chunk_shape_2d, divs_2d)
        
        # Check that regions 1, 2, 3 are merged (should become one label)
        merged_mask = (regions_2d_2 == 1) | (regions_2d_2 == 2) | (regions_2d_2 == 3)
        assert np.unique(stitched_regions_2d_2[merged_mask]).size == 1
        # Check that region 4 remains separate and has a different label
        assert np.unique(stitched_regions_2d_2[regions_2d_2 == 4]).size == 1
        assert np.unique(stitched_regions_2d_2[merged_mask])[0] != np.unique(stitched_regions_2d_2[regions_2d_2 == 4])[0]
        # Check that there are exactly 2 unique labels (1 merged group + 1 isolated)
        assert np.unique(stitched_regions_2d_2[stitched_regions_2d_2 > 0]).size == 2
        
        # Test case 3: A 3D image, 2x1x1 divisions, labels should merge across Z axis boundary
        regions_3d = np.zeros((4, 2, 2), dtype=np.int32)
        regions_3d[0:2, :, :] = 1 # First chunk (0,0,0) with label 1
        regions_3d[2:4, :, :] = 2 # Second chunk (1,0,0) with label 2
        chunk_shape_3d = (2, 2, 2)
        divs_3d = (2, 1, 1)

        stitched_regions_3d = _stitch_chunks_variable(regions_3d, chunk_shape_3d, divs_3d)
        
        # All non-zero labels should merge into a single label
        assert np.unique(stitched_regions_3d[regions_3d > 0]).size == 1
        assert np.all(stitched_regions_3d[regions_3d > 0] == stitched_regions_3d[1, 0, 0])




if __name__ == '__main__':
    t = SnowsTest()
    t.setup_class()
    self = t
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
