import cv2
import numpy as np

import tensorflow as tf

from monopsr.core import transform_utils
from monopsr.datasets.kitti import calib_utils, depth_map_utils, obj_utils


def read_instance_image(instance_image_path):
    instance_image = cv2.imread(instance_image_path, cv2.IMREAD_GRAYSCALE)
    return instance_image


def get_instance_image(sample_name, instance_dir):

    instance_image_path = instance_dir + '/{}.png'.format(sample_name)
    instance_image = read_instance_image(instance_image_path)

    return instance_image


def get_instance_mask_list(instance_img, num_instances=None):
    """Creates n-dimensional image from instance image with one channel per instance

    Args:
        instance_img: (H, W) instance image
        num_instances: (optional) number of instances in the image. If None, will use the
            highest value pixel as the number of instances, but may miss the last
            instances if they have no points.

    Returns:
        instance_masks: (k, H, W) instance masks where k is the unique values of the instance im
    """

    if num_instances is None:
        valid_pixels = instance_img[instance_img != 255]
        if len(valid_pixels) == 0:
            return []
        num_instances = np.max(valid_pixels) + 1

    instance_masks = np.asarray([(instance_img == instance_idx)
                                 for instance_idx in range(num_instances)])
    return instance_masks


def read_instance_maps(instance_maps_path):
    return np.load(instance_maps_path)


def get_valid_inst_box_2d_crop(box_2d, input_map):
    """Gets a valid 2D box crop. If the box is too small, it returns a single pixel.

    Args:
        box_2d: 2D box
        input_map: (H, W, C) Input map

    Returns:
        inst_box_2d_crop: Crop of input map
    """

    # Get box dimensions
    box_2d_rounded = np.round(box_2d).astype(np.int32)
    box_2d_rounded_h = box_2d_rounded[2] - box_2d_rounded[0]
    box_2d_rounded_w = box_2d_rounded[3] - box_2d_rounded[1]

    # Check dimensions
    if box_2d_rounded_h > 0 and box_2d_rounded_w > 0:
        # Crop if valid box
        inst_box_2d_crop = input_map[box_2d_rounded[0]:box_2d_rounded[2],
                                     box_2d_rounded[1]:box_2d_rounded[3]]
    else:
        # Invalid box, use single pixel
        inst_box_2d_crop = input_map[
            box_2d_rounded[0]:box_2d_rounded[0] + 1,
            box_2d_rounded[1]:box_2d_rounded[1] + 1]

    return inst_box_2d_crop


def np_instance_crop(boxes_2d, boxes_3d, instance_masks, input_map, roi_size,
                     view_norm=False, cam_p=None, viewing_angles=None,
                     centroid_type='bottom', rotate_view=True):
    """Crops an input map for an instance

    Args:
        boxes_2d: (N, 4) 2D boxes [y1, x1, y2, x2]
        boxes_3d: (N, 6) 3D boxes
        instance_masks:  (N, H, W) boolean instance masks
        input_map: (H, W, C) Input map with C channels. Should be in camN frame.
        roi_size: roi crop size [h, w]
        view_norm: (optional) Apply view normalization for xyz maps
        cam_p: (3, 4) Camera projection matrix
        viewing_angles: (N) Viewing angles
        centroid_type (string): centroid position (bottom or middle)
        rotate_view: bool whether to rotating by viewing angle

    Returns:
        all_instance_xyz: (N, roi_h, roi_w, C) cropped and resized instance map
        valid_pixel_mask: (N, roi_h, roi_w, 1) mask of valid pixels
    """
    # TODO: Add unit tests, fix valid pixel mask return

    input_map_shape = input_map.shape

    if len(input_map_shape) != 3:
        raise ValueError('Invalid input_map_shape', input_map_shape)

    all_instance_maps = []
    all_valid_mask_maps = []
    for instance_idx, (instance_mask, box_2d, box_3d) in enumerate(
            zip(instance_masks, boxes_2d, boxes_3d)):

        # Apply instance mask
        input_map_masked = instance_mask[:, :, np.newaxis] * input_map

        # Crop and resize
        inst_box_2d_crop = get_valid_inst_box_2d_crop(box_2d, input_map_masked)
        instance_map_resized = cv2.resize(inst_box_2d_crop, tuple(roi_size),
                                          interpolation=cv2.INTER_NEAREST)

        # Calculate valid mask, works for both point clouds and RGB
        instance_map_resized_shape = instance_map_resized.shape
        if len(instance_map_resized_shape) == 3:
            valid_mask_map = np.sum(abs(instance_map_resized), axis=2) > 0.1
        else:
            valid_mask_map = abs(instance_map_resized) > 0.1
        all_valid_mask_maps.append(valid_mask_map)

        if view_norm:
            if input_map.shape[2] != 3:
                raise ValueError('Invalid shape to apply view normalization')

            # Get viewing angle and rotation matrix
            viewing_angle = viewing_angles[instance_idx]

            # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
            x_offset = -cam_p[0, 3] / cam_p[0, 0]
            cam0_centroid = box_3d[0:3]
            camN_centroid = cam0_centroid - [x_offset, 0, 0]

            if centroid_type == 'middle':
                # Move centroid to half the box height
                half_h = box_3d[5] / 2.0
                camN_centroid[1] -= half_h

            if rotate_view:
                inst_xyz_map_local = apply_view_norm_to_pc_map(
                    instance_map_resized, valid_mask_map, viewing_angle, camN_centroid,
                    roi_size)
            else:
                inst_xyz_map_local = apply_view_norm_to_pc_map(
                    instance_map_resized, valid_mask_map, 0.0, camN_centroid, roi_size)

            all_instance_maps.append(inst_xyz_map_local)

        else:
            all_instance_maps.append(instance_map_resized)

    return np.asarray(all_instance_maps), np.asarray(all_valid_mask_maps)


def np_instance_xyz_crop(boxes_2d, boxes_3d, instance_masks, xyz_map, roi_size,
                         view_norm=False, cam_p=None, viewing_angles=None,
                         centroid_type='bottom', rotate_view=True):
    """Crops an input map for an instance

    Args:
        boxes_2d: (N, 4) 2D boxes [y1, x1, y2, x2]
        boxes_3d: (N, 6) 3D boxes
        instance_masks:  (N, H, W) boolean instance masks
        xyz_map: (H, W, C) Input map with C channels. Should be in camN frame.
        roi_size: roi crop size [h, w]
        view_norm: (optional) Apply view normalization for xyz maps
        cam_p: (3, 4) Camera projection matrix
        viewing_angles: (N) Viewing angles
        centroid_type (string): centroid position (bottom or middle)
        rotate_view: bool whether to rotating by viewing angle

    Returns:
        all_instance_xyz: (N, roi_h, roi_w, C) cropped and resized instance map
        valid_pixel_mask: (N, roi_h, roi_w, 1) mask of valid pixels
    """
    # TODO: Add unit tests, fix valid pixel mask return

    input_map_shape = xyz_map.shape

    if len(input_map_shape) != 3:
        raise ValueError('Invalid input_map_shape', input_map_shape)

    all_instance_maps = []
    all_valid_mask_maps = []
    for instance_idx, (instance_mask, box_2d, box_3d) in enumerate(
            zip(instance_masks, boxes_2d, boxes_3d)):

        # Apply instance mask
        input_map_masked = instance_mask[:, :, np.newaxis] * xyz_map

        # Crop and resize
        inst_box_2d_crop = get_valid_inst_box_2d_crop(box_2d, input_map_masked)
        instance_map_resized = cv2.resize(inst_box_2d_crop, tuple(roi_size),
                                          interpolation=cv2.INTER_NEAREST)

        # Calculate valid mask, works for both point clouds and RGB
        instance_map_resized_shape = instance_map_resized.shape
        if len(instance_map_resized_shape) == 3:
            valid_mask_map = np.sum(abs(instance_map_resized), axis=2) > 0.1
        else:
            valid_mask_map = abs(instance_map_resized) > 0.1
        all_valid_mask_maps.append(valid_mask_map)

        if view_norm:
            if xyz_map.shape[2] != 3:
                raise ValueError('Invalid shape to apply view normalization')

            # Get viewing angle and rotation matrix
            viewing_angle = viewing_angles[instance_idx]

            # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
            x_offset = -cam_p[0, 3] / cam_p[0, 0]
            cam0_centroid = box_3d[0:3]
            camN_centroid = cam0_centroid - [x_offset, 0, 0]

            if centroid_type == 'middle':
                # Move centroid to half the box height
                half_h = box_3d[5] / 2.0
                camN_centroid[1] -= half_h

            inst_pc_map = instance_map_resized.transpose([2, 0, 1])

            if rotate_view:
                inst_xyz_map_local = apply_view_norm_to_pc_map(
                    inst_pc_map, valid_mask_map, viewing_angle, camN_centroid,
                    roi_size)
            else:
                inst_xyz_map_local = apply_view_norm_to_pc_map(
                    inst_pc_map, valid_mask_map, 0.0, camN_centroid, roi_size)

            all_instance_maps.append(inst_xyz_map_local)

        else:
            all_instance_maps.append(instance_map_resized)

    return np.asarray(all_instance_maps), np.asarray(all_valid_mask_maps)


def tf_instance_xyz_crop(box_idx,
                         tf_boxes_2d,
                         tf_boxes_3d,
                         tf_instance_masks,
                         tf_xyz_map_batched,
                         roi_size,
                         tf_viewing_angles,
                         cam_p,
                         centroid_type='bottom',
                         rotate_view=True):
    """Crops and rotates the xyz map for an instance

    Args:
        box_idx: box index
        tf_boxes_2d: (N, 4) 2D boxes [y1, x1, y2, x2]
        tf_boxes_3d: (N, 6) 3D boxes
        tf_instance_masks: (N, H, W) boolean instance masks
        tf_xyz_map_batched: (1, H, W, 3) xyz map
        roi_size: [h, w] roi crop size
        cam_p: (3, 4) Camera projection matrix
        tf_viewing_angles: (N) viewing angles
        centroid_type (string): centroid position (bottom or middle)
        rotate_view: bool whether to rotate by viewing angle

    Returns:
        xyz_out: (N, roi_h, roi_w, 3) instance xyz map
        valid_pixel_mask: (N, roi_h, roi_w, 1) mask of valid pixels
    """
    # TODO: Add other representations (e.g. depth or distance)

    with tf.variable_scope('crop_{}'.format(box_idx)):
        box_2d = tf_boxes_2d[box_idx]
        box_2d_rounded = tf.to_int32(tf.round(box_2d))
        instance_mask = tf_instance_masks[box_idx]

        xyz_masked = tf_xyz_map_batched * tf.expand_dims(instance_mask, axis=2)
        xyz_cropped = xyz_masked[:,
                                 box_2d_rounded[0]:box_2d_rounded[2],
                                 box_2d_rounded[1]:box_2d_rounded[3]]
        xyz_resized = tf.image.resize_nearest_neighbor(
            xyz_cropped, roi_size, align_corners=True)

        # Get viewing angle rotation matrix
        viewing_angle = tf_viewing_angles[box_idx]
        cam0_centroid = tf_boxes_3d[box_idx, 0:3]

        # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
        x_offset = -cam_p[0, 3] / cam_p[0, 0]
        camN_centroid = cam0_centroid - [x_offset, 0, 0]

        if centroid_type == 'middle':
            # Move centroid to half the box height
            half_h = tf_boxes_3d[box_idx, 5] / 2.0
            camN_centroid = camN_centroid - [0, half_h, 0]

        if rotate_view:
            tr_mat = transform_utils.tf_get_tr_mat(-viewing_angle, -camN_centroid)
        else:
            tr_mat = tf.to_float([
                [1.0, 0.0, 0.0, -camN_centroid[0]],
                [0.0, 1.0, 0.0, -camN_centroid[1]],
                [0.0, 0.0, 1.0, -camN_centroid[2]],
                [0.0, 0.0, 0.0, 1.0],
            ])

        # Pad for matrix multiplication
        pc_resized = tf.transpose(tf.reshape(xyz_resized, [-1, 3]))
        pc_padded = tf.pad(pc_resized, [[0, 1], [0, 0]], constant_values=1.0)

        # Transform into local space
        xyz_local = tf.transpose(tf.matmul(tr_mat, pc_padded)[0:3])
        xyz_local = tf.reshape(xyz_local, (1, *roi_size, 3))

        # Calculate valid pixel mask
        valid_pixel_mask = tf.reduce_max(
            tf.to_float(tf.greater_equal(tf.abs(xyz_resized), 0.1)), axis=3, keepdims=True)

        # Only keep valid pixels
        xyz_out = xyz_local * valid_pixel_mask

    # return xyz_masked, xyz_resized, xyz_normalized
    return tf.stop_gradient(xyz_out), valid_pixel_mask


def np_instance_xyz_crop_from_depth_map(boxes_2d, boxes_3d, instance_masks,
                                        depth_map, roi_size, cam_p, viewing_angles,
                                        use_pixel_centres, use_corr_factors, centroid_type='bottom',
                                        rotate_view=True):
    """Crops the depth map for an instance and returns local instance xyz crops

    Args:
        boxes_2d: (N, 4) List of 2D boxes [y1, x1, y2, x2]
        boxes_3d: (N, 6) 3D boxes
        instance_masks: (N, H, W) Boolean instance masks
        depth_map: (H, W) Depth map
        roi_size: ROI crop size [h, w]
        cam_p: (3, 4) Camera projection matrix
        viewing_angles: (N) Viewing angles
        use_pixel_centres: (optional) If True, re-projects depths such that they will
            project back to the centre of the ROI pixel. Otherwise, they will project
            to the top left corner.
        use_corr_factors: (optional) If True, applies correction factors along xx and yy
            according to depth in order to reduce projection error.
        centroid_type (string): centroid position (bottom or middle)
        rotate_view: bool whether to rotate by viewing angle

    Returns:
        xyz_out: (N, roi_h, roi_w, 3) instance xyz map in local coordinate frame
        valid_pixel_mask: (N, roi_h, roi_w, 1) mask of valid pixels
    """

    depth_map_shape = depth_map.shape
    if len(depth_map_shape) != 2:
        raise ValueError('Invalid depth_map_shape', depth_map_shape)

    all_inst_depth_crops, all_inst_valid_masks = np_instance_crop(
        boxes_2d=boxes_2d,
        boxes_3d=boxes_3d,
        instance_masks=instance_masks,
        input_map=np.expand_dims(depth_map, 2),
        roi_size=roi_size,
        view_norm=False)

    camN_inst_pc_maps = [depth_map_utils.depth_patch_to_pc_map(
        inst_depth_crop, box_2d, cam_p, roi_size, depth_map_shape=depth_map.shape[0:2],
        use_pixel_centres=use_pixel_centres, use_corr_factors=use_corr_factors)
        for inst_depth_crop, box_2d in zip(all_inst_depth_crops, boxes_2d)]

    # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
    x_offset = -cam_p[0, 3] / cam_p[0, 0]
    camN_centroids = boxes_3d[:, 0:3] - [x_offset, 0, 0]

    if centroid_type == 'middle':
        # Move centroid to half the box height
        half_h = boxes_3d[:, 5] / 2.0
        camN_centroids[:, 1] -= half_h

    if not rotate_view:
        viewing_angles = np.zeros_like(viewing_angles)

    inst_xyz_maps_local = [
        apply_view_norm_to_pc_map(inst_pc_map, valid_mask, viewing_angle, centroid, roi_size)
        for inst_pc_map, valid_mask, viewing_angle, centroid in zip(
            camN_inst_pc_maps, all_inst_valid_masks, viewing_angles, camN_centroids)]

    return inst_xyz_maps_local, all_inst_valid_masks


def tf_instance_xyz_crop_from_depth_map(box_idx, tf_boxes_2d, tf_boxes_3d, tf_instance_masks,
                                        tf_depth_map_batched, roi_size, tf_viewing_angles, cam_p,
                                        view_norm=False, centroid_type='bottom', rotate_view=True):
    """Crops the depth map for an instance and returns local instance xyz crops

    Args:
        box_idx: box index
        tf_boxes_2d: (N, 4) 2D boxes [y1, x1, y2, x2]
        tf_boxes_3d: (N, 6) 3D boxes
        tf_instance_masks: (N, H, W) boolean instance masks
        tf_depth_map_batched: (1, H, W, 1) depth map
        roi_size: [h, w] roi crop size
        tf_viewing_angles: (N) viewing angles
        cam_p: (3, 4) Camera projection matrix
        view_norm: bool whether to perform any view normalization
        centroid_type (string): centroid position (bottom or middle)
        rotate_view: bool whether to rotate by viewing angle

    Returns:
        xyz_out: (N, roi_h, roi_w, 3) instance xyz map in local coordinate frame
        valid_pixel_mask: (N, roi_h, roi_w, 1) mask of valid pixels
    """
    # TODO: Add other representations (e.g. depth or distance)

    with tf.variable_scope('crop_{}'.format(box_idx)):
        box_2d = tf_boxes_2d[box_idx]
        box_2d_rounded = tf.to_int32(tf.round(box_2d))
        instance_mask = tf_instance_masks[box_idx]

        depth_map_masked = tf_depth_map_batched * tf.expand_dims(instance_mask, axis=2)
        depth_map_cropped = depth_map_masked[:,
                                             box_2d_rounded[0]:box_2d_rounded[2],
                                             box_2d_rounded[1]:box_2d_rounded[3]]
        depth_map_resized = tf.image.resize_nearest_neighbor(
            depth_map_cropped, roi_size, align_corners=True)

        # Convert to xyz map
        inst_pc_map = depth_map_utils.tf_depth_patch_to_pc_map(
            depth_map_resized, box_2d, cam_p, roi_size, use_pixel_centres=True)

        # Calculate valid pixel mask
        valid_pixel_mask = tf.reduce_max(
            tf.to_float(tf.greater_equal(tf.abs(depth_map_resized), 0.1)), axis=3, keepdims=True)

        if view_norm:
            # Get viewing angle rotation matrix
            viewing_angle = tf_viewing_angles[box_idx]
            cam0_centroid = tf_boxes_3d[box_idx, 0:3]

            # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
            x_offset = -cam_p[0, 3] / cam_p[0, 0]
            camN_centroid = cam0_centroid - [x_offset, 0, 0]

            if centroid_type == 'middle':
                # Move centroid to half the box height
                half_h = tf_boxes_3d[box_idx, 5] / 2.0
                camN_centroid = camN_centroid - [0, half_h, 0]

            if rotate_view:
                tr_mat = transform_utils.tf_get_tr_mat(-viewing_angle, -camN_centroid)
            else:
                tr_mat = tf.to_float([
                    [1.0, 0.0, 0.0, -camN_centroid[0]],
                    [0.0, 1.0, 0.0, -camN_centroid[1]],
                    [0.0, 0.0, 1.0, -camN_centroid[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ])

            # Pad for matrix multiplication
            # pc_resized = tf.transpose(tf.reshape(inst_pc_map, [-1, 3]))
            pc_resized = tf.reshape(inst_pc_map, [3, -1])
            pc_padded = tf.pad(pc_resized, [[0, 1], [0, 0]], constant_values=1.0)

            # Transform into local space
            xyz_local = tf.transpose(tf.matmul(tr_mat, pc_padded)[0:3])
            xyz_local = tf.reshape(xyz_local, (1, *roi_size, 3))

            # Only keep valid pixels
            xyz_out = xyz_local * valid_pixel_mask

        else:
            # TODO: Directly reshape?
            pc = tf.transpose(tf.reshape(inst_pc_map, [3, -1]))
            xyz_out = tf.reshape(pc, (1, *roi_size, 3)) * valid_pixel_mask

    # return xyz_masked, xyz_resized, xyz_normalized
    return tf.stop_gradient(xyz_out), valid_pixel_mask


def tf_crop_and_resize_instance_masks(instance_masks, boxes_2d, roi_size, box_idx):
    """Crops and resize instance masks using nearest neighbor

    Args:
        instance_masks: (N, H, W) instance mask
        boxes_2d: (N, 4) box 2d
        roi_size: ROI size [h, w]
        box_idx: Box index

    Returns:
        instance_maps_cropped_resized: instance mask cropped and resized
    """
    box_2d = boxes_2d[box_idx]
    box_2d_rounded = tf.to_int32(tf.round(box_2d))
    instance_mask = instance_masks[box_idx]

    instance_masks_batched = tf.expand_dims(instance_mask, axis=0)

    instance_masks_cropped = instance_masks_batched[:,
                                                    box_2d_rounded[0]: box_2d_rounded[2],
                                                    box_2d_rounded[1]: box_2d_rounded[3]]

    instance_maps_cropped_resized = tf.image.resize_nearest_neighbor(
        tf.expand_dims(instance_masks_cropped, axis=3), roi_size, align_corners=True)

    return instance_maps_cropped_resized


def apply_view_norm_to_pc_map(inst_pc_map, valid_mask_map, viewing_angle, centroid, roi_size):
    """Applies view normalization on instance pc map

    Args:
        inst_pc_map: (3, H, W) Instance pc map
        valid_mask_map: (H, W) Valid pixel mask
        viewing_angle: Viewing angle
        centroid: Centroid [x, y, z]
        roi_size: ROI size [h, w]

    Returns:
        inst_xyz_map: (H, W, 3) View normalized xyz map
    """

    # Apply view normalization
    tr_mat = transform_utils.np_get_tr_mat(-viewing_angle, -centroid)

    # Move to origin
    inst_pc_padded = transform_utils.pad_pc(inst_pc_map.reshape(3, -1))
    inst_pc_local = tr_mat.dot(inst_pc_padded)[0:3]

    inst_xyz_map = np.reshape(inst_pc_local.T, (*roi_size, 3))
    inst_xyz_map = inst_xyz_map * np.expand_dims(valid_mask_map, 2)

    return inst_xyz_map


def inst_points_global_to_local(inst_points_global, viewing_angle, centroid):
    """Converts global points to local points in same camera frame."""

    # Apply view normalization
    rot_mat = transform_utils.np_get_tr_mat(-viewing_angle, -centroid)

    # Rotate, then translate
    inst_pc_padded = transform_utils.pad_pc(inst_points_global.T)
    inst_pc_local = rot_mat.dot(inst_pc_padded)[0:3]

    return inst_pc_local.T


def inst_points_local_to_global(inst_points_local, viewing_angle, centroid):
    """Converts local points to global points in same camera frame"""

    # Rotate predicted instance points to viewing angle and translate to guessed centroid
    rot_mat = transform_utils.np_get_tr_mat(viewing_angle, (0.0, 0.0, 0.0))
    t_mat = transform_utils.np_get_tr_mat(0.0, centroid)

    inst_points_rotated = transform_utils.apply_tr_mat_to_points(
        rot_mat, inst_points_local)

    inst_points_global = transform_utils.apply_tr_mat_to_points(t_mat, inst_points_rotated)

    return inst_points_global


def tf_inst_xyz_map_local_to_global(inst_xyz_map_local, map_roi_size,
                                    view_angs, centroids):
    """Converts a local instance xyz map to a global instance xyz map

    Args:
        inst_xyz_map_local: (N, H, W, 3) Local instance xyz map
        map_roi_size: Map ROI size
        view_angs: Viewing angles
        centroids: (N, 3) Centroids

    Returns:
        inst_xyz_map_global: (N, H, W, 3) Global instance xyz map
    """

    num_inst = inst_xyz_map_local.shape[0]
    map_roi_h = map_roi_size[0]
    map_roi_w = map_roi_size[1]

    # Convert prediction maps to point cloud format
    inst_pc_map_local = tf.transpose(inst_xyz_map_local, [0, 3, 1, 2])
    inst_pc_local = tf.reshape(
        inst_pc_map_local, [num_inst, 3, map_roi_h * map_roi_w])

    # Transform to predicted global position
    rot_mat, _, _ = transform_utils.tf_get_tr_mat_batch(view_angs, tf.zeros_like(centroids))
    t_mat, _, _ = transform_utils.tf_get_tr_mat_batch(tf.zeros_like(view_angs), centroids)
    # tr_mat, rot_mat, t_mat = transform_utils.tf_get_tr_mat_batch(view_angs, centroids)
    inst_pc_local_padded = transform_utils.tf_pad_pc(inst_pc_local)

    inst_pc_rotated = tf.matmul(rot_mat, inst_pc_local_padded)
    inst_pc_global_padded = tf.matmul(t_mat, inst_pc_rotated)

    # Convert to global xyz map
    inst_pc_map_global = tf.reshape(inst_pc_global_padded[:, 0:3],
                                    [num_inst, 3, map_roi_h, map_roi_w])
    inst_xyz_map_global = tf.transpose(inst_pc_map_global, [0, 2, 3, 1])

    return inst_xyz_map_global


def tf_inst_depth_map_local_to_global(inst_depth_map_local, global_depth,
                                      box_2d=None, inst_view_ang=None,
                                      map_roi_size=None, cam_p=None, rotate_view=False):
    """Converts a local instance depth map to a global instance depth map by adding a constant
    to it. Optionally performs rotation to undo view normalization. See view_normalization_depth.py

    Args:
        inst_depth_map_local: Local depth map [N, H, W, 1]
        global_depth: Scalar of how much to scale local depth by
        box_2d: Box 2D of instance
        inst_view_ang: Viewing angle of instance
        map_roi_size: [H, W]
        cam_p: camera projection matrix
        rotate_view: Bool whether to undo view normalization

    Returns:
        inst_depth_map_global: Global depth map [N, H, W, 1]

    """

    if rotate_view:

        centre_u = cam_p[0, 2]
        focal_length = cam_p[0, 0]

        # Use 2D box left and right edges
        box_x1 = box_2d[:, 1]
        box_x2 = box_2d[:, 3]

        # Account for pixel centres
        grid_spacing = (box_x2 - box_x1) / map_roi_size[0] / 2.0
        box_x1 += grid_spacing
        box_x2 -= grid_spacing

        # Assume depth of 1.0 to calculate viewing angle
        # viewing_angle = atan2(i / f, 1.0)
        view_ang_l = tf.atan2((box_x1 - centre_u) / focal_length, 1.0)
        view_ang_l = tf.expand_dims(view_ang_l, axis=1)
        view_ang_r = tf.atan2((box_x2 - centre_u) / focal_length, 1.0)
        view_ang_r = tf.expand_dims(view_ang_r, axis=1)

        inst_xz = global_depth / tf.cos(inst_view_ang)

        # Calculate depth offset for each edge
        l_o = inst_xz / tf.cos(view_ang_l - inst_view_ang)
        r_o = inst_xz / tf.cos(view_ang_r - inst_view_ang)

        x_l = l_o * tf.sin(view_ang_l - inst_view_ang)
        x_r = r_o * tf.sin(view_ang_r - inst_view_ang)

        offset_l = tf.squeeze(x_l * tf.sin(inst_view_ang))
        offset_r = tf.squeeze(x_r * tf.sin(inst_view_ang))

        # Linearly interpolate across
        view_ang_depth_offset = \
            tf.map_fn(lambda x: tf.linspace(x[0], x[1], map_roi_size[0]),
                      (-offset_l, -offset_r), dtype=tf.float32)

        # Reshape for broadcasting
        pred_cen_z_reshaped = tf.reshape(global_depth, [-1, 1, 1, 1])
        view_ang_depth_offset_reshaped = \
            tf.tile(tf.reshape(view_ang_depth_offset, (-1, map_roi_size[0], 1, 1)),
                    (1, 1, 48, 1))

        # Get global point cloud
        inst_depth_map_global = \
            (inst_depth_map_local + pred_cen_z_reshaped + view_ang_depth_offset_reshaped)
    else:
        # Reshape for broadcasting
        pred_cen_z_reshaped = tf.reshape(global_depth, [-1, 1, 1, 1])

        # Add predicted centroid depth to get global depth
        inst_depth_map_global = inst_depth_map_local + pred_cen_z_reshaped

    return inst_depth_map_global


def get_exp_proj_uv_map(box_2d, roi_size, round_box_2d=False, use_pixel_centres=False):
    """Get expected grid projection of a 2D box based on roi size, if pixels are evenly spaced.
    Points project to the top left of each pixel.

    Args:
        box_2d: 2D box
        roi_size: ROI size [h, w]
        use_pixel_centres: (optional) If True, return projections to centre of pixels

    Returns:
        proj_uv_map: (H, W, 2) Expected box_2d projection uv map
    """

    # Grid start and stop
    if round_box_2d:
        inst_u1, inst_u2 = np.round(box_2d[[1, 3]])
        inst_v1, inst_v2 = np.round(box_2d[[0, 2]])
    else:
        inst_u1, inst_u2 = box_2d[[1, 3]]
        inst_v1, inst_v2 = box_2d[[0, 2]]

    # Grid spacing
    roi_h, roi_w = roi_size
    grid_u_spacing = (inst_u2 - inst_u1) / roi_w
    grid_v_spacing = (inst_v2 - inst_v1) / roi_h

    if use_pixel_centres:

        # Grid along u
        grid_u_half_spacing = grid_u_spacing / 2.0
        grid_u = np.linspace(
            inst_u1 + grid_u_half_spacing,
            inst_u2 - grid_u_half_spacing,
            roi_w)

        # Grid along v
        grid_v_half_spacing = grid_v_spacing / 2.0
        grid_v = np.linspace(
            inst_v1 + grid_v_half_spacing,
            inst_v2 - grid_v_half_spacing,
            roi_h)

        proj_uv_map = np.meshgrid(grid_u, grid_v)

    else:
        # Use linspace instead of arange to avoid including last value
        grid_u = np.linspace(inst_u1, inst_u2 - grid_u_spacing, roi_w)
        grid_v = np.linspace(inst_v1, inst_v2 - grid_v_spacing, roi_h)

        proj_uv_map = np.meshgrid(grid_u, grid_v)

    return np.dstack(proj_uv_map)


def tf_get_exp_proj_uv_map(tf_boxes_2d, roi_size,
                           round_box_2d=False,
                           use_pixel_centres=True):
    """Calculates expected

    Args:
        tf_boxes_2d: (N, 4) Tensor of 2D boxes
        roi_size: ROI size [h, w]
        round_box_2d (bool): Optional, whether to round the 2D boxes
        use_pixel_centres (boo;): Optional, whether to use centre of pixels

    Returns:
        exp_proj_uv_map: (N, H, W, 2) Expected UV projection map
    """

    if round_box_2d:
        tf_boxes_2d = tf.round(tf_boxes_2d)

    # Grid start and stop
    inst_v1 = tf_boxes_2d[:, 0]
    inst_u1 = tf_boxes_2d[:, 1]
    inst_v2 = tf_boxes_2d[:, 2]
    inst_u2 = tf_boxes_2d[:, 3]

    # Grid spacing
    roi_h, roi_w = roi_size
    grid_u_spacing = (inst_u2 - inst_u1) / roi_w
    grid_v_spacing = (inst_v2 - inst_v1) / roi_h

    if use_pixel_centres:
        # Calculate half grid spacing
        grid_u_half_spacing = grid_u_spacing / 2.0
        grid_v_half_spacing = grid_v_spacing / 2.0

        grid_u = tf.map_fn(lambda x: tf.linspace(x[0], x[1], roi_size[0]),
                           (inst_u1 + grid_u_half_spacing,
                            inst_u2 - grid_u_half_spacing), dtype=tf.float32)
        grid_v = tf.map_fn(lambda x: tf.linspace(x[0], x[1], roi_size[1]),
                           (inst_v1 + grid_v_half_spacing,
                            inst_v2 - grid_v_half_spacing), dtype=tf.float32)
    else:
        grid_u = tf.map_fn(lambda x: tf.linspace(x[0], x[1], roi_size[0]),
                           (inst_u1, inst_u2 - grid_u_spacing), dtype=tf.float32)
        grid_v = tf.map_fn(lambda x: tf.linspace(x[0], x[1], roi_size[1]),
                           (inst_v1, inst_v2 - grid_u_spacing), dtype=tf.float32)

    exp_proj_uv_map = tf.stack(
        tf.map_fn(lambda x: tf.meshgrid(x[0], x[1]), [grid_u, grid_v]),
        axis=3)

    return exp_proj_uv_map


def proj_points(xz_dist, centroid_y, viewing_angle, cam2_inst_points_local,
                cam_p, rotate_view=True):
    """Projects point based on estimated transformation matrix
    calculated from xz_dist and viewing angle

    Args:
        xz_dist: distance along viewing angle
        centroid_y: box centroid y
        viewing_angle: viewing angle
        cam2_inst_points_local: (N, 3) instance points
        cam_p: (3, 4) camera projection matrix
        rotate_view: bool whether to rotate by viewing angle

    Returns:
        points_uv: (2, N) The points projected in u, v coordinates
        valid_points_mask: (N) Mask of valid points
    """

    guess_x = xz_dist * np.sin(viewing_angle)
    guess_y = centroid_y
    guess_z = xz_dist * np.cos(viewing_angle)

    # Rotate predicted instance points to viewing angle and translate to guessed centroid
    rot_mat = transform_utils.np_get_tr_mat(viewing_angle, (0.0, 0.0, 0.0))
    t_mat = transform_utils.np_get_tr_mat(0.0, [guess_x, guess_y, guess_z])
    if rotate_view:
        cam2_points_rotated = transform_utils.apply_tr_mat_to_points(
            rot_mat, cam2_inst_points_local)
    else:
        cam2_points_rotated = cam2_inst_points_local

    cam2_points_global = transform_utils.apply_tr_mat_to_points(t_mat, cam2_points_rotated)

    # Get valid points mask
    valid_points_mask = np.sum(np.abs(cam2_points_rotated), axis=1) > 0.1

    # Shift points into cam0 frame for projection
    # Get x offset (b_cam) from calibration: cam_mat[0, 3] = (-f_x * b_cam)
    x_offset = -cam_p[0, 3] / cam_p[0, 0]

    # Shift points from cam2 to cam0 frame
    cam0_points_global = (cam2_points_global + [x_offset, 0, 0]) * valid_points_mask.reshape(-1, 1)

    # Project back to image
    pred_points_in_img = calib_utils.project_pc_to_image(
        cam0_points_global.T, cam_p) * valid_points_mask

    return pred_points_in_img, valid_points_mask


def est_y_from_box_2d_and_depth(cam_p, box_2d, depth, centroid_type,
                                obj_h=None, class_str=None, trend_data='kitti'):

    # Parse camera projection matrix
    focal_length = cam_p[0, 0]
    centre_v = cam_p[1, 2]

    # Estimate y by projecting the (u,v) coordinates of the box_2d centre
    box_2d_centre_v = (box_2d[2] + box_2d[0]) / 2. - centre_v
    z = depth

    cen_y_mid_estimate = box_2d_centre_v * (z / focal_length)

    if centroid_type == 'middle':
        # Calculate and subtract average offset (see compare_y_estimate.py)

        if class_str == 'Car':
            if trend_data == 'kitti':
                y_estimate = cen_y_mid_estimate - 0.0648
            elif trend_data == 'mscnn':
                y_estimate = cen_y_mid_estimate - 0.0655

        elif class_str == 'Pedestrian':
            if trend_data == 'kitti':
                y_estimate = cen_y_mid_estimate - 0.0145
            elif trend_data == 'mscnn':
                y_estimate = cen_y_mid_estimate - 0.0142

        elif class_str == 'Cyclist':
            if trend_data == 'kitti':
                y_estimate = cen_y_mid_estimate - 0.0239
            elif trend_data == 'mscnn':
                y_estimate = cen_y_mid_estimate - 0.0239

        else:
            raise ValueError('Invalid class_str', class_str)

    elif centroid_type == 'bottom':
        if obj_h is None:
            # Use average height
            obj_h = obj_utils.MEAN_HEIGHTS[class_str]

            # Calculate and subtract average offset (see compare_y_estimate.py)
            # y_estimate = box_2d_centre_v * (z / focal_length) + (obj_h / 2.0)

            if trend_data == 'kitti':
                y_estimate = box_2d_centre_v * (z / focal_length) + (obj_h / 2.0) - 0.0641
            elif trend_data == 'mscnn':
                y_estimate = box_2d_centre_v * (z / focal_length) + (obj_h / 2.0) - 0.0637

        else:
            # Calculate and subtract average offset (see compare_y_estimate.py)
            # y_estimate = box_2d_centre_v * (z / focal_length) + (obj_h / 2.0)
            y_estimate = cen_y_mid_estimate + (obj_h / 2.0) - 0.0648

            if trend_data == 'kitti':
                y_estimate = box_2d_centre_v * (z / focal_length) + (obj_h / 2.0) - 0.0648
            elif trend_data == 'mscnn':
                y_estimate = box_2d_centre_v * (z / focal_length) + (obj_h / 2.0) - 0.0655

    else:
        raise ValueError('Invalid centroid type', centroid_type)

    return y_estimate


def tf_est_y_from_box_2d_and_depth(cam_p, box_2d, depth,
                                   class_str=None, trend_data='kitti'):
    """Estimates the y centroid position by projecting the centre of the 2D box

    Args:
        cam_p: camera projection matrix
        box_2d: [N, 4] box_2d
        depth: [N, 1] depth of the instance
        class_str: str of the object class
        trend_data: Data source used to determine offsets/trends

    Returns:
        y_estimate: [N, 1] estimate of the y centroid position
    """

    # Parse camera projection matrix
    focal_length = cam_p[0, 0]
    centre_v = cam_p[1, 2]

    # Estimate y by projecting the (u,v) coordinates of the box_2d centre
    box_2d_centre_v = tf.expand_dims((box_2d[:, 2] + box_2d[:, 0]) / 2. - centre_v, axis=1)
    z = depth
    cen_y_mid_estimate = box_2d_centre_v * (z / focal_length)

    # Subtract average offset
    if class_str == 'Car':
        if trend_data == 'kitti':
            y_estimate = cen_y_mid_estimate - 0.0648
        elif trend_data == 'mscnn':
            y_estimate = cen_y_mid_estimate - 0.0655

    elif class_str == 'Pedestrian':
        if trend_data == 'kitti':
            y_estimate = cen_y_mid_estimate - 0.0145
        elif trend_data == 'mscnn':
            y_estimate = cen_y_mid_estimate - 0.0142

    elif class_str == 'Cyclist':
        if trend_data == 'kitti':
            y_estimate = cen_y_mid_estimate - 0.0239
        elif trend_data == 'mscnn':
            y_estimate = cen_y_mid_estimate - 0.0239

    else:
        raise ValueError('Invalid class_str', class_str)

    return y_estimate


def est_y_from_box_2d_and_xz_dist(cam_p, box_2d, viewing_angle, xz_dist, obj_type):

    # Parse camera projection matrix
    focal_length = cam_p[0, 0]
    centre_v = cam_p[1, 2]

    # Estimate y by projecting the (u,v) coordinates of the box_2d centre
    box_2d_centre_v = (box_2d[2] + box_2d[0]) / 2. - centre_v
    z = xz_dist * np.cos(viewing_angle)

    avg_box_3d_h = obj_utils.MEAN_HEIGHTS[obj_type]
    y_estimate = box_2d_centre_v * (z / focal_length) + (avg_box_3d_h / 2.0)

    return y_estimate


def get_prop_cen_z_offset(class_str):
    """Get the proposal z centroid offset depending on the class.
    """

    if class_str == 'Car':
        offset = 2.17799973487854
    elif class_str == 'Pedestrian':
        offset = 0.351921409368515
    elif class_str == 'Cyclist':
        offset = 0.8944902420043945
    else:
        raise ValueError('Invalid class_str', class_str)

    return offset


def postprocess_cen_x(pred_box_2d, pred_box_3d, cam_p):
    """Post-process centroid x by projecting predicted 3D box and finding width ratio to centre,
    and then finding the u position by using this ratio on the 2D box and projecting it.

    Args:
        pred_box_2d: 2D box in box_2d format [y1, x1, y2, x2]
        pred_box_3d: 3D box in box_3d format [x, y, z, l, w, h, ry]
        cam_p: camera projection matrix

    Returns:
        new_cen_x: post-processed centroid x position
    """

    focal_length = cam_p[0, 0]
    centre_u = cam_p[0, 2]

    # Project corners
    pred_box_3d_corners = obj_utils.compute_box_3d_corners(pred_box_3d)
    pred_box_corners_uv = calib_utils.project_pc_to_image(pred_box_3d_corners, cam_p)

    # Project centroid
    pred_cen_pc = pred_box_3d[0:3, np.newaxis]
    pred_cen_uv = calib_utils.project_pc_to_image(pred_cen_pc, cam_p)

    # Find min u
    pred_box_min_u = np.amin(pred_box_corners_uv[0])
    pred_box_max_u = np.amax(pred_box_corners_uv[0])

    # Find centroid u ratio
    pred_box_w = pred_box_max_u - pred_box_min_u
    pred_box_cen_u_ratio = (pred_cen_uv[0] - pred_box_min_u) / pred_box_w

    # Find new u from original 2D detection
    pred_box_w = pred_box_2d[3] - pred_box_2d[1]
    pred_box_u = pred_box_2d[1] + pred_box_cen_u_ratio * pred_box_w

    # Calculate new centroid x
    i = pred_box_u - centre_u

    # Similar triangles ratio (x/i = d/f)
    pred_cen_z = pred_box_3d[2]
    ratio = pred_cen_z / focal_length
    new_cen_x = i * ratio

    return new_cen_x
