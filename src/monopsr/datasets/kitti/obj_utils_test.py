import numpy as np
import tensorflow as tf

from monopsr.builders.dataset_builder import DatasetBuilder
from monopsr.datasets.kitti import obj_utils


class ObjUtilsTest(tf.test.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAINVAL)

    def test_filter_obj_labels(self):

        sample_name = '000050'
        obj_labels = obj_utils.read_labels(self.dataset.kitti_label_dir, sample_name)

        obj_labels_filt, obj_mask = obj_utils.filter_labels(
            obj_labels, classes=['Car'], depth_range=[5, 45])

        self.assertTrue(len(obj_labels_filt) == 3)
        np.testing.assert_equal(obj_mask, [True, True, False, True, False])

    def test_tf_centre_box_2d_coordinates(self):
        """Test conversion from pixel coordinates to film coordinates
        """
        boxes_2d = [
            [1.0, 2.0, 7.0, 8.0],
            [7.0, 8.0, 9.0, 10.0],
        ]
        cam_p = np.full([3, 4], 6.0)

        exp_boxes_2d_ij = [
            [-5.0, -4.0, 1.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
        ]

        tf_boxes_2d = tf.to_float(boxes_2d)
        tf_cam_p = tf.to_float(cam_p)
        tf_boxes_2d_ij = obj_utils.tf_boxes_2d_ij_fmt(tf_boxes_2d, tf_cam_p)

        with self.test_session() as sess:
            boxes_2d_ij_out = sess.run(tf_boxes_2d_ij)

        np.testing.assert_allclose(boxes_2d_ij_out, exp_boxes_2d_ij)
