import numpy as np
import tensorflow as tf

from monopsr.datasets.kitti import feature_normalization


class ObjUtilsTest(tf.test.TestCase):

    def test_tf_normalize_cen_y_by_mean(self):
        """Test normalizing cen y by mean
        """
        tf_obj_class = tf.expand_dims(tf.constant(['Car', 'Pedestrian', 'Cyclist']), 1)
        unnormalized_cen_y = tf.expand_dims(tf.constant([1.0, 1.0, 1.0]), 1)

        # See box_means.py
        avg_car_cen_y = 1.7153475
        avg_ped_cen_y = 1.4557862
        avg_cyc_cen_y = 1.5591882

        tf_normalized_cen_y = feature_normalization.tf_normalize_cen_y_by_mean(unnormalized_cen_y,
                                                                               tf_obj_class)

        with self.test_session() as sess:
            normalized_cen_y = sess.run(tf_normalized_cen_y)

        np.testing.assert_allclose(np.squeeze(normalized_cen_y),
                                   [1. / avg_car_cen_y, 1. / avg_ped_cen_y, 1. / avg_cyc_cen_y])
