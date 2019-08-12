
import numpy as np
import tensorflow as tf


def np_get_tr_mat(ry, t):
    """Calculates a transformation matrix in numpy (translation followed by rotation)

    Args:
        ry: Rotation along the y-axis
        t: Translation [x, y, z]

    Returns:
        tr_mat: Transformation matrix
    """

    rot_mat = np.asarray([
        [np.cos(ry), 0.0, np.sin(ry), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(ry), 0, np.cos(ry), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    t_mat = np.asarray([
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])

    tr_mat = np.matmul(rot_mat, t_mat)

    return tr_mat


def tf_get_tr_mat(ry, t):
    """Calculates a transformation matrix in Tensorflow

    Args:
        ry: Rotation along the y-axis
        t: Translation [x, y, z]

    Returns:
        tr_mat: Transformation matrix
    """

    cos_ry = tf.cos(ry)
    sin_ry = tf.sin(ry)

    rot_mat = tf.to_float([
        [cos_ry, 0.0, sin_ry, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sin_ry, 0.0, cos_ry, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    t_mat = tf.to_float([
        [1.0, 0.0, 0.0, t[0]],
        [0.0, 1.0, 0.0, t[1]],
        [0.0, 0.0, 1.0, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ])

    tr_mat = tf.matmul(rot_mat, t_mat)

    return tr_mat


def tf_get_tr_mat_batch(ry, t):
    """Calculates a batch of transform matrices

    Args:
        ry: (N, 1) Rotation around y
        t: (N, 3) Translation

    Returns:
        tr_mat: (N, 4, 4) Transform matrices
        rot_mat: (N, 4, 4) Rotation matrices
        t_mat: (N, 4, 4) Translation matrices
    """

    # Remove last axis
    ry = tf.squeeze(ry, axis=1)

    batch_size = ry.shape[0]
    zeros = tf.zeros([batch_size])
    ones = tf.ones([batch_size])

    cos_ry = tf.cos(ry)
    sin_ry = tf.sin(ry)

    rot_mat = tf.stack([
        tf.stack([cos_ry, zeros, sin_ry, zeros], axis=1),
        tf.stack([zeros, ones, zeros, zeros], axis=1),
        tf.stack([-sin_ry, zeros, cos_ry, zeros], axis=1),
        tf.stack([zeros, zeros, zeros, ones], axis=1),
    ], axis=1)

    t_mat = tf.stack([
        tf.stack([ones, zeros, zeros, t[:, 0]], axis=1),
        tf.stack([zeros, ones, zeros, t[:, 1]], axis=1),
        tf.stack([zeros, zeros, ones, t[:, 2]], axis=1),
        tf.stack([zeros, zeros, zeros, ones], axis=1),
    ], axis=1)

    tr_mat = tf.matmul(rot_mat, t_mat)

    return tr_mat, rot_mat, t_mat


# TODO: Rename to apply_tf_mat_to_points
def apply_tr_mat_to_points(tf_mat, points):
    """Applies a transformation to a set of points

    Args:
        tf_mat: (4, 4) Transformation matrix
        points: (N, 3) List of points

    Returns:
        points_transformed: (N, 3) Transformed points
    """

    pc_padded = pad_points(points).T
    pc_transformed = tf_mat.dot(pc_padded)
    points_transformed = pc_transformed[0:3].T

    return points_transformed


def invert_tf(tf_matrix):
    """Inverts a transformation matrix

    Args:
        tf_matrix: (4, 4) Transformation matrix

    Returns:
        mat_inv: (4, 4) Matrix inverse
    """
    # rot_inv = np.linalg.inv(matrix[0:3, 0:3])
    rot_inv = tf_matrix[0:3, 0:3].T
    t_inv = -tf_matrix[0:3, 3]

    mat_inv = np.zeros((4, 4), np.float32)
    mat_inv[0:3, 0:3] = rot_inv
    mat_inv[0:3, 3] = np.dot(rot_inv, t_inv)
    mat_inv[3, 3] = 1.0

    return mat_inv


def pad_pc(point_cloud):
    """Pads a point cloud

    Args:
        point_cloud: (3, N) Point cloud

    Returns:
        point_cloud_padded: (4, N) Padded point cloud
    """
    return np.pad(point_cloud, ((0, 1), (0, 0)), constant_values=1.0, mode='constant')


def tf_pad_pc(point_cloud):

    """Pads a point cloud

    Args:
        point_cloud: (B, 3, N) Point cloud

    Returns:
        point_cloud_padded: (B, 4, N) Padded point cloud
    """
    return tf.pad(point_cloud, ((0, 0), (0, 1), (0, 0)), constant_values=1.0)


def pad_points(points):
    """Pads a set of points

    Args:
        points: (N, 3) List of points

    Returns:
        points_padded: (N, 4) Padded points
    """
    return np.pad(points, ((0, 0), (0, 1)), constant_values=1.0, mode='constant')
