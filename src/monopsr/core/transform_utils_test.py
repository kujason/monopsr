import numpy as np
import tensorflow as tf

from monopsr.core import transform_utils


class TransformUtilsTest(tf.test.TestCase):

    def test_np_get_tr_mat(self):

        ry_0 = 0.0
        t_0 = np.zeros(3)

        ry_90 = np.deg2rad(90.0)
        t_246 = [2.0, 4.0, 6.0]

        # Rotation: 0, translation: variable
        tr_mat_0_0 = transform_utils.np_get_tr_mat(ry_0, t_0)
        tr_mat_0_246 = transform_utils.np_get_tr_mat(ry_0, t_246)

        # Rotation: 90, translation: 0
        tr_mat_90_0 = transform_utils.np_get_tr_mat(ry_90, t_0)

        # Check for expected values
        np.testing.assert_allclose(tr_mat_0_0, np.eye(4))

        exp_0_246_mat = np.eye(4)
        exp_0_246_mat[0:3, 3] = [2.0, 4.0, 6.0]
        np.testing.assert_allclose(tr_mat_0_246, exp_0_246_mat)

        exp_90_0_mat = np.eye(4)
        exp_90_0_mat[0:3, 0:3] = [
            [+0.0, +0.0, +1.0],
            [+0.0, +1.0, +0.0],
            [-1.0, +0.0, +0.0],
        ]
        np.testing.assert_allclose(tr_mat_90_0, exp_90_0_mat, atol=1E-7)

    def test_tf_get_tr_mat(self):

        ry_0 = tf.to_float(0.0)
        t_0 = tf.to_float(np.zeros(3, dtype=np.float32))
        ry_90 = tf.to_float(np.deg2rad(90.0))
        t_246 = tf.to_float([2.0, 4.0, 6.0])

        # Rotation: 0, translation: variable
        tr_mat_0_0 = transform_utils.tf_get_tr_mat(ry_0, t_0)
        tr_mat_0_246 = transform_utils.tf_get_tr_mat(ry_0, t_246)

        # Rotation: 90, translation: 0
        tr_mat_90_0 = transform_utils.tf_get_tr_mat(ry_90, t_0)

        with self.test_session() as sess:
            tr_mat_0_0_out, tr_mat_0_246_out, tr_mat_90_0_out = sess.run(
                [tr_mat_0_0, tr_mat_0_246, tr_mat_90_0])

            # Check for expected values
            np.testing.assert_allclose(tr_mat_0_0_out, np.eye(4))

            exp_0_246_mat = np.eye(4)
            exp_0_246_mat[0:3, 3] = [2.0, 4.0, 6.0]
            np.testing.assert_allclose(tr_mat_0_246_out, exp_0_246_mat)

            exp_90_0_mat = np.eye(4)
            exp_90_0_mat[0:3, 0:3] = [
                [+0.0, +0.0, +1.0],
                [+0.0, +1.0, +0.0],
                [-1.0, +0.0, +0.0],
            ]
            np.testing.assert_allclose(tr_mat_90_0_out, exp_90_0_mat, atol=1E-7)

    def test_tf_get_tr_mat_batched(self):
        """Check batched transform matrix generation"""

        batch_size = 1

        view_angs = np.linspace(0.0, np.pi, batch_size).astype(np.float32)
        t_0 = np.zeros([batch_size, 3], dtype=np.float32)
        t_246 = np.reshape(np.tile([2.0, 4.0, 6.0], [batch_size]), [batch_size, 3])

        tf_view_angs = tf.reshape(view_angs, [batch_size, 1])
        tf_t_0 = tf.to_float(t_0)
        tf_t_246 = tf.to_float(t_246)

        # Rotation: 0, translation: variable
        tr_mat_0, rot_mat_0, t_mat_0 = transform_utils.tf_get_tr_mat_batch(tf_view_angs, tf_t_0)
        tr_mat_246, rot_mat_246, t_mat_246 = transform_utils.tf_get_tr_mat_batch(
            tf_view_angs, tf_t_246)

        exp_tr_mat_0 = [transform_utils.np_get_tr_mat(view_ang, t)
                        for view_ang, t in zip(view_angs, t_0)]
        exp_tr_mat_246 = [transform_utils.np_get_tr_mat(view_ang, t)
                          for view_ang, t in zip(view_angs, t_246)]

        with self.test_session() as sess:
            tr_mat_0_out, tr_mat_246_out = sess.run([tr_mat_0, tr_mat_246])

        # Compare with expected values from numpy version
        np.testing.assert_allclose(tr_mat_0_out, exp_tr_mat_0)
        np.testing.assert_allclose(tr_mat_246_out, exp_tr_mat_246)
