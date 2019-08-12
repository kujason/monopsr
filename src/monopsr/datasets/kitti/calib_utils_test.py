import unittest

import numpy as np
import tensorflow as tf
from monopsr.builders.dataset_builder import DatasetBuilder

from monopsr.datasets.kitti import instance_utils, calib_utils, obj_utils


class InstanceUtilsTest(tf.test.TestCase):

    def test_tf_project_pc_to_image(self):
        """Check that tf_project_pc_to_image matches numpy version"""

        dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAINVAL)

        np.random.seed(12345)
        point_cloud_batch = np.random.rand(32, 3, 2304)
        frame_calib = calib_utils.get_frame_calib(dataset.calib_dir, '000050')
        cam_p = frame_calib.p2

        exp_proj_uv = [calib_utils.project_pc_to_image(point_cloud, cam_p)
                       for point_cloud in point_cloud_batch]

        tf_proj_uv = calib_utils.tf_project_pc_to_image(point_cloud_batch, cam_p, 32)

        with self.test_session() as sess:
            proj_uv_out = sess.run(tf_proj_uv)

        np.testing.assert_allclose(exp_proj_uv, proj_uv_out)
