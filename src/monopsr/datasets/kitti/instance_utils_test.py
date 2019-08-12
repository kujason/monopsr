import unittest

import numpy as np
import tensorflow as tf

from monopsr.datasets.kitti import instance_utils


class InstanceUtilsTest(tf.test.TestCase):

    def test_get_proj_uv_map(self):

        box_2d = np.asarray([0, 10, 10, 20], dtype=np.float32)
        roi_size = (10, 10)

        proj_uv_map = instance_utils.get_exp_proj_uv_map(box_2d, roi_size)
        proj_u_row = proj_uv_map[0][0]
        proj_v_col = proj_uv_map[1][:, 0]

        # Check that points are in the middle of each pixel
        exp_u_row = np.linspace(10.5, 19.5, 10)
        exp_v_row = np.linspace(0.5, 9.5, 10)

        np.testing.assert_allclose(proj_u_row, exp_u_row)
        np.testing.assert_allclose(proj_v_col, exp_v_row)

    def test_tf_get_proj_uv_map(self):

        boxes_2d = np.asarray([
            [0.0, 10.0, 10.0, 20.0],
            [5.0, 5.0, 10.0, 10.0],
            [0.0, 0.0, 100.0, 100.0],
        ], np.float32)

        roi_size = (10, 10)

        exp_proj_uv_maps = [instance_utils.get_exp_proj_uv_map(
            box_2d, roi_size, use_pixel_centres=True)
            for box_2d in boxes_2d]

        # Convert to tensors
        tf_boxes_2d = tf.to_float(boxes_2d)

        proj_uv_map = instance_utils.tf_get_exp_proj_uv_map(tf_boxes_2d, roi_size)

        with self.test_session() as sess:
            proj_uv_map_out = sess.run(proj_uv_map)

        # Compare with expected
        np.testing.assert_allclose(proj_uv_map_out, exp_proj_uv_maps)

    def test_tf_inst_xyz_map_local_to_global(self):

        inst_points_local = np.random.rand(2304, 3).astype(np.float32)
        viewing_angle = np.deg2rad(10.0).astype(np.float32)
        centroid = np.asarray([2.5, 1.5, 15.0], dtype=np.float32)

        np_inst_points_global = instance_utils.inst_points_local_to_global(
            inst_points_local, viewing_angle, centroid)

        xyz_maps_local = inst_points_local.reshape(1, 48, 48, 3)
        tf_view_angs = np.reshape(viewing_angle, (-1, 1))
        tf_centroids = np.reshape(centroid, (-1, 3))
        tf_inst_xyz_map_global = instance_utils.tf_inst_xyz_map_local_to_global(
            xyz_maps_local, map_roi_size=(48, 48),
            view_angs=tf_view_angs, centroids=tf_centroids)

        with self.test_session() as sess:
            tf_inst_xyz_map_global_out = sess.run(tf_inst_xyz_map_global)

        # Check equivalence
        tf_inst_points_global = tf_inst_xyz_map_global_out.reshape(2304, 3)
        np.testing.assert_allclose(np_inst_points_global, tf_inst_points_global)

