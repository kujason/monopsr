import copy

import numpy as np

from monopsr.core import evaluation
from monopsr.datasets.kitti import kitti_aug, obj_utils

AUG_FLIPPING = 'flipping'
AUG_PCA_JITTER = 'pca_jitter'


def flip_image(image):
    """Flips an image horizontally
    """
    flipped_image = np.fliplr(image)
    return flipped_image


def flip_points(points):
    """Flips a list of points (N, 3)
    """
    flipped_points = np.copy(points)
    flipped_points[:, 0] = -points[:, 0]
    return flipped_points


def flip_point_cloud(point_cloud):
    """Flips a point cloud (3, N)
    """
    flipped_point_cloud = np.copy(point_cloud)
    flipped_point_cloud[0] = -point_cloud[0]
    return flipped_point_cloud


def flip_label_in_3d_only(obj_label):
    """Flips only the 3D position of an object label. The 2D bounding box is
    not flipped to save time since it is not used.

    Args:
        obj_label: ObjectLabel

    Returns:
        A flipped object
    """

    flipped_label = copy.deepcopy(obj_label)

    # Flip the rotation
    if obj_label.ry >= 0:
        flipped_label.ry = np.pi - obj_label.ry
    else:
        flipped_label.ry = -np.pi - obj_label.ry

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_t = (-flipped_label.t[0], flipped_label.t[1], flipped_label.t[2])
    flipped_label.t = flipped_t

    return flipped_label


def flip_boxes_3d(boxes_3d, flip_ry=True):
    """Flips boxes_3d

    Args:
        boxes_3d: List of boxes in box_3d format
        flip_ry bool: (optional) if False, rotation is not flipped to save on
            computation (useful for flipping anchors)

    Returns:
        flipped_boxes_3d: Flipped boxes in box_3d format
    """

    flipped_boxes_3d = np.copy(boxes_3d)

    if flip_ry:
        # Flip the rotation
        above_zero = boxes_3d[:, 6] >= 0
        below_zero = np.logical_not(above_zero)
        flipped_boxes_3d[above_zero, 6] = np.pi - boxes_3d[above_zero, 6]
        flipped_boxes_3d[below_zero, 6] = -np.pi - boxes_3d[below_zero, 6]

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_boxes_3d[:, 0] = -boxes_3d[:, 0]

    return flipped_boxes_3d


def flip_ground_plane(ground_plane):
    """Flips the ground plane by negating the x coefficient
        (ax + by + cz + d = 0)

    Args:
        ground_plane: ground plane coefficients

    Returns:
        Flipped ground plane coefficients
    """
    flipped_ground_plane = np.copy(ground_plane)
    flipped_ground_plane[0] = -ground_plane[0]
    return flipped_ground_plane


def flip_stereo_calib_p2(calib_p2, image_shape):
    """Flips the stereo calibration matrix to correct the projection back to
    image space. Flipping the image can be seen as a movement of both the
    camera plane, and the camera itself. To account for this, the instrinsic
    matrix x0 value is flipped with respect to the image width, and the
    extrinsic matrix t1 value is negated.

    Args:
        calib_p2: 3 x 4 stereo camera calibration matrix
        image_shape: (h, w) image shape

    Returns:
        'Flipped' calibration p2 matrix with shape (3, 4)
    """
    flipped_p2 = np.copy(calib_p2)
    flipped_p2[0, 2] = image_shape[1] - calib_p2[0, 2]
    flipped_p2[0, 3] = -calib_p2[0, 3]

    return flipped_p2


def apply_image_noise(image_rgb):
    """Applies PCA jitter or random noise to a single image

    Args:
        image_rgb: RGB image to modify

    Returns:
        Modified image
    """
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    image_out = image_rgb

    # Random value
    random_values = np.random.rand(5)

    # Swap B and G channels in RGB
    if random_values[0] < 0.10:
        image_out = np.copy(image_rgb)
        image_out[:, :, 1], image_out[:, :, 2] = \
            image_out[:, :, 2], image_out[:, :, 1]

    # Gaussian noise
    if random_values[1] < 0.40:
        gaussian_noise = np.random.randn(*image_rgb.shape) * 10.0
        image_out = np.uint8(
            np.clip(image_rgb + gaussian_noise, 0.0, 255.0))

    # Channel specific noise
    if random_values[2] < 0.40:
        channel_gaussian_noise = np.random.randn(3) * 8.0
        image_out = np.uint8(
            np.clip(image_rgb + channel_gaussian_noise, 0.0, 255.0))

    # Brightness
    if random_values[3] < 0.40:
        random_brightness = np.random.randn(1) * 15.0
        image_out = np.uint8(
            np.clip(image_rgb + random_brightness, 0.0, 255.0))

    # Random uniform noise
    if random_values[4] < 0.40:
        random_amount = np.random.uniform(0, 10)
        random_noise = np.random.uniform(-random_amount, random_amount, image_rgb.shape)
        image_out = np.uint8(
            np.clip(image_rgb + random_noise, 0.0, 255.0))

    return image_out


def jitter_obj_boxes_2d(obj_labels, iou_threshold_min, image_shape):
    """Jitters 2D bounding boxes.
    This is computed in a brute force fashion. For any given
    bounding box, we randomly shift the centroid and generate
    new bounding boxes and if it lies above the desired IoU
    threshold bound, we will keep it. Otherwise it is thrown
    out and this is repeated until a bounding box that meets
    the desired iou_threshold_min is created. Also ensures that
    the new 2d boxes are within the dimensions of the image shape.

    Args:
        obj_labels (list): labels of shape (N,)
        iou_threshold_min: minimum IoU between required between
            the original box and the generated boxes
        image_shape: shape of the image

    Returns:
        new_objs (ndarray): obj_labels with modified box coordinates
    """

    img_height = image_shape[0]
    img_width = image_shape[1]

    new_objs = []
    for obj_label in obj_labels:

        # Parse original box coordinates
        x1 = obj_label.x1
        y1 = obj_label.y1
        x2 = obj_label.x2
        y2 = obj_label.y2
        original_box = np.asarray([[x1, y1, x2, y2]])

        # Compute half the width and length and centroid
        box_w = (x2 - x1)
        box_h = (y2 - y1)
        half_w = box_w / 2
        half_h = box_h / 2
        centroid_x = (x2 + x1) / 2
        centroid_y = (y2 + y1) / 2

        # Create a copy of the label
        new_obj = copy.deepcopy(obj_label)

        # If the box is less than 10 pixels in width or height, avoid jittering
        if box_w < 10 or box_h < 10:
            new_objs.append(new_obj)
        else:
            # Generate boxes until a bounding box achieves the IoU requirement
            iou = 0
            while iou < iou_threshold_min:

                # Determine new centroids
                new_centroid_x = np.random.normal(centroid_x, half_w / 3)
                new_centroid_y = np.random.normal(centroid_y, half_h / 3)

                # Alter box size
                new_half_w = np.random.normal(half_w, half_w / 6)
                new_half_h = np.random.normal(half_h, half_h / 6)

                # Determine new coordinates. Ensure coordinates don't reach outside the image
                # dimensions
                new_x1 = np.maximum(0, (new_centroid_x - new_half_w))
                new_x2 = np.minimum(img_width - 1, (new_centroid_x + new_half_w))
                new_y1 = np.maximum(0, (new_centroid_y - new_half_h))
                new_y2 = np.minimum(img_height - 1, (new_centroid_y + new_half_h))

                new_box = np.asarray([new_x1, new_y1, new_x2, new_y2])

                # Calculate the IoU
                iou = evaluation.two_d_iou(new_box, original_box)

            # Create new obj with jittered box
            new_obj.x1 = new_x1
            new_obj.y1 = new_y1
            new_obj.x2 = new_x2
            new_obj.y2 = new_y2
            new_objs.append(new_obj)

    new_objs = np.asarray(new_objs)

    return new_objs
