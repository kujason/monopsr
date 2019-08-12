"""
This module converts data to and from the 'box_3d' format
 [x, y, z, l, w, h, ry]
"""
import numpy as np
import tensorflow as tf

from monopsr.core import format_checker as fc
from monopsr.datasets.kitti import obj_utils


def box_3d_to_object_label(box_3d, obj_type='Car'):
    """Turns a box_3d into an ObjectLabel

    Args:
        box_3d: 3D box in the format [x, y, z, l, w, h, ry]
        obj_type: Optional, the object type

    Returns:
        ObjectLabel with the location, size, and rotation filled out
    """

    fc.check_box_3d_format(box_3d)

    obj_label = obj_utils.ObjectLabel()

    obj_label.type = obj_type

    obj_label.t = box_3d.take((0, 1, 2))
    obj_label.l = box_3d[3]
    obj_label.w = box_3d[4]
    obj_label.h = box_3d[5]
    obj_label.ry = box_3d[6]

    return obj_label


def object_label_to_box_2d(obj_label):
    """Converts an ObjectLabel into box_2d

    Args:
        obj_label: ObjectLabel to convert

    Returns:
        box_2d: 2D box in box_2d format [y1, x1, y2, x2]
    """
    fc.check_obj_label_format(obj_label)
    box_2d = np.asarray([obj_label.y1, obj_label.x1, obj_label.y2, obj_label.x2], np.float32)
    return box_2d


def object_label_to_box_3d(obj_label):
    """Turns an ObjectLabel into a box_3d

    Args:
        obj_label: ObjectLabel

    Returns:
        box_3d: 3D box in box_3d format [x, y, z, l, w, h, ry]
    """

    fc.check_obj_label_format(obj_label)

    box_3d = np.zeros(7, dtype=np.float32)

    box_3d[0:3] = obj_label.t
    box_3d[3:6] = obj_label.l, obj_label.w, obj_label.h
    box_3d[6] = obj_label.ry

    return box_3d


def boxes_2d_to_iou_fmt(boxes_2d):
    """Converts a list of boxes_2d [y1, x1, y2, x2] to
    iou format [x1, y1, x2, y2]"""
    boxes_2d = np.asarray(boxes_2d)
    boxes_2d_iou_fmt = boxes_2d[:, [1, 0, 3, 2]]
    return boxes_2d_iou_fmt


# TODO Remove dependency on this function
def box_3d_to_3d_iou_format(boxes_3d):
    """ Returns a numpy array of 3d box format for iou calculation
    Args:
        boxes_3d: list of 3d boxes
    Returns:
        new_anchor_list: numpy array of 3d box format for iou
    """
    boxes_3d = np.asarray(boxes_3d)
    fc.check_box_3d_format(boxes_3d)

    iou_3d_boxes = np.zeros([len(boxes_3d), 7])
    iou_3d_boxes[:, 4:7] = boxes_3d[:, 0:3]
    iou_3d_boxes[:, 1] = boxes_3d[:, 3]
    iou_3d_boxes[:, 2] = boxes_3d[:, 4]
    iou_3d_boxes[:, 3] = boxes_3d[:, 5]
    iou_3d_boxes[:, 0] = boxes_3d[:, 6]

    return iou_3d_boxes


def tf_box_3d_diagonal_length(boxes_3d):
    """Returns the diagonal length of box_3d

    Args:
        boxes_3d: An tensor of shape (N x 7) of boxes in box_3d format.

    Returns:
        Diagonal of all boxes, a tensor of (N,) shape.
    """

    lengths_sqr = tf.square(boxes_3d[:, 3])
    width_sqr = tf.square(boxes_3d[:, 4])
    height_sqr = tf.square(boxes_3d[:, 5])

    lwh_sqr_sums = lengths_sqr + width_sqr + height_sqr
    diagonals = tf.sqrt(lwh_sqr_sums)

    return diagonals


def compute_box_3d_corners(box_3d):
    """Computes 3D corners from a box_3d

    Args:
        box_3d: (N, 7) 3D boxes

    Returns:
        corners_3d: array of box corners 8 x [x y z]
    """

    tx, ty, tz, l, w, h, ry = box_3d

    half_l = l / 2
    half_w = w / 2

    # Compute rotation matrix
    rot = np.array([[+np.cos(ry), 0, +np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, +np.cos(ry)]])

    # 3D BB corners
    x_corners = np.array(
        [half_l, half_l, -half_l, -half_l, half_l, half_l, -half_l, -half_l])
    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])
    z_corners = np.array(
        [half_w, -half_w, -half_w, half_w, half_w, -half_w, -half_w, half_w])
    corners_3d = np.dot(rot, np.array([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + tx
    corners_3d[1, :] = corners_3d[1, :] + ty
    corners_3d[2, :] = corners_3d[2, :] + tz

    return np.asarray(corners_3d, dtype=np.float32)
