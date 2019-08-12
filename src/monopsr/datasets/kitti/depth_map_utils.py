import cv2
import numpy as np
import png
import tensorflow as tf

from monopsr.datasets.kitti import calib_utils


def read_depth_map(depth_map_path):

    depth_image = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
    depth_map = depth_image / 256.0

    # Discard depths less than 10cm from the camera
    depth_map[depth_map < 0.1] = 0.0

    return depth_map.astype(np.float32)


def save_depth_map(save_path, depth_map,
                   version='cv2', png_compression=3):
    """Saves depth map to disk as uint16 png

    Args:
        save_path: path to save depth map
        depth_map: depth map numpy array [h w]
        version: 'cv2' or 'pypng'
        png_compression: Only when version is 'cv2', sets png compression level.
            A lower value is faster with larger output,
            a higher value is slower with smaller output.
    """

    # Convert depth map to a uint16 png
    depth_image = (depth_map * 256.0).astype(np.uint16)

    if version == 'cv2':
        cv2.imwrite(save_path, depth_image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])

    elif version == 'pypng':
        with open(save_path, 'wb') as f:
            depth_image = (depth_map * 256.0).astype(np.uint16)
            writer = png.Writer(width=depth_image.shape[1],
                                height=depth_image.shape[0],
                                bitdepth=16,
                                greyscale=True)
            writer.write(f, depth_image)

    else:
        raise ValueError('Invalid version', version)


def depth_patch_to_pc_map(depth_patch, box_2d, cam_p, roi_size,
                          round_box_2d=True, use_pixel_centres=True,
                          use_corr_factors=True, depth_map_shape=None):
    """Calculates a point cloud map of the desired ROI size from a patch of
    a depth map given the camera parameters.

    Args:
        depth_patch: patch of a depth map
        box_2d: 2D box corresponding to the depth patch
        cam_p: camera p matrix
        roi_size: ROI size (h, w)
        round_box_2d: (optional) Whether to round the 2D box
        use_pixel_centres: (optional) If True, re-projects depths such that they will
            project back to the centre of the ROI pixel. Otherwise, they will project
            to the top left corner.
        use_corr_factors: (optional) If True, applies correction factors along xx and yy
            according to depth in order to reduce projection error.
        depth_map_shape (optional): Original depth map shape, required if using correction factors

    Returns:
        point_cloud: (3, N) point cloud
    """

    if round_box_2d:
        y1, x1, y2, x2 = np.round(box_2d)
    else:
        y1, x1, y2, x2 = box_2d

    num_roi_pixels_x = roi_size[0]
    num_roi_pixels_y = roi_size[1]

    roi_pixel_w = (x2 - x1) / num_roi_pixels_x
    roi_pixel_h = (y2 - y1) / num_roi_pixels_y

    # Use box_2d for meshgrid
    if use_pixel_centres:

        half_pixel_w = roi_pixel_w / 2.0
        half_pixel_h = roi_pixel_h / 2.0

        xx, yy = np.meshgrid(
            np.linspace(x1 + half_pixel_w, x2 - half_pixel_w, num_roi_pixels_x),
            np.linspace(y1 + half_pixel_h, y2 - half_pixel_h, num_roi_pixels_y))

    else:
        xx, yy = np.meshgrid(
            np.linspace(x1, x2 - roi_pixel_w, num_roi_pixels_x),
            np.linspace(y1, y2 - roi_pixel_h, num_roi_pixels_y))

    if use_corr_factors:
        _apply_corr_factor(depth_patch, depth_map_shape, xx, yy)

    # Parse camera projection matrix
    focal_length = cam_p[0, 0]
    centre_u = cam_p[0, 2]
    centre_v = cam_p[1, 2]

    i = xx - centre_u
    j = yy - centre_v

    # Similar triangles ratio (x/i = d/f)
    ratio = depth_patch / focal_length

    x = i * ratio
    y = j * ratio
    z = depth_patch

    # TODO: add in_cam0_frame
    # x_offset = -cam_p[0, 3] / focal_length
    # point = np.asarray([x + x_offset, y, z])

    pc_map = np.asarray((x, y, z))

    return pc_map


def _apply_corr_factor(depths, depth_map_shape, xx, yy):
    """Overwrites xx and yy with values with new values calculated with
    a correction factor based on depth.

    Args:
        depths: Depths
        depth_map_shape: Depth map shape
        xx: Original meshgrid xx
        yy: Original meshgrid yy
    """
    depth_map_h, depth_map_w = depth_map_shape

    valid_mask = depths > 0.1
    valid_depths = depths[valid_mask]

    # Correction factors for better projection (see calc_corr_factor_for_depth_to_xyz)
    if depth_map_w == 1242:
        xx_offset = np.clip(3.38 * (valid_depths ** -0.998), 0.049, 0.68)
        yy_offset = np.clip(0.729 * (valid_depths ** -0.998), 0.0105, 0.146)

    elif depth_map_w == 1224:
        xx_offset = np.clip(6.07 * (valid_depths ** -1.0), 0.087, 1.22)
        yy_offset = np.clip(2.30 * (valid_depths ** -1.0), 0.033, 0.459)

    else:
        # TODO: add other correction factor
        raise NotImplementedError('depth_map_w not supported yet', depth_map_w)

    # Overwrite values
    xx[valid_mask] += xx_offset * (xx[valid_mask] / (depth_map_w))
    yy[valid_mask] += yy_offset * (yy[valid_mask] / (depth_map_h))


def tf_depth_patch_to_pc_map(depth_patch, box_2d, cam_p, roi_size,
                             use_pixel_centres=True,
                             use_corr_factors=False, depth_map_shape=None):
    """Calculates a point cloud map, in the camera frame, of the desired ROI size from a patch of
    a depth map given the camera parameters.

    Args:
        depth_patch: patch of a depth map
        box_2d: 2D box corresponding to the depth patch
        cam_p: camera p matrix
        roi_size: ROI size (h, w)
        use_pixel_centres: (optional) If True, re-projects depths such that they will
            project back to the centre of the ROI pixel. Otherwise, they will project
            to the top left corner.
        use_corr_factors: (optional) If True, applies correction factors along xx and yy
            according to depth in order to reduce projection error.
        depth_map_shape: (optional) Original depth map shape, required if using correction factors

    Returns:
        point_cloud: (3, N) point cloud
    """

    y1 = box_2d[0]
    x1 = box_2d[1]
    y2 = box_2d[2]
    x2 = box_2d[3]

    num_roi_pixels_x = roi_size[0]
    num_roi_pixels_y = roi_size[1]

    roi_pixel_w = (x2 - x1) / num_roi_pixels_x
    roi_pixel_h = (y2 - y1) / num_roi_pixels_y

    # Use box_2d for meshgrid
    if use_pixel_centres:

        half_pixel_w = roi_pixel_w / 2.0
        half_pixel_h = roi_pixel_h / 2.0

        xx, yy = tf.meshgrid(
            tf.linspace(x1 + half_pixel_w, x2 - half_pixel_w, num_roi_pixels_x),
            tf.linspace(y1 + half_pixel_h, y2 - half_pixel_h, num_roi_pixels_y))

    else:
        xx, yy = tf.meshgrid(
            tf.linspace(x1, x2 - roi_pixel_w, num_roi_pixels_x),
            tf.linspace(y1, y2 - roi_pixel_h, num_roi_pixels_y))

    # TODO: Implement correction factors
    # if use_corr_factors:
    #     _apply_corr_factor(depth_patch, depth_map_shape, xx, yy)

    # Parse camera projection matrix
    focal_length = cam_p[0, 0]
    centre_u = cam_p[0, 2]
    centre_v = cam_p[1, 2]

    i = xx - centre_u
    j = yy - centre_v

    depth_patch_squeezed = tf.squeeze(depth_patch)

    # Similar triangles ratio (x/i = d/f)
    ratio = depth_patch_squeezed / focal_length

    x = i * ratio
    y = j * ratio
    z = depth_patch_squeezed

    # TODO: add in_cam0_frame
    # x_offset = -cam_p[0, 3] / focal_length
    # point = np.asarray([x + x_offset, y, z])

    pc_map = tf.stack((x, y, z), axis=0)

    return pc_map


def get_depth_point_cloud(depth_map, cam_p, min_v=0, flatten=True, in_cam0_frame=True,
                          use_corr_factors=False):
    """Calculates the point cloud from a depth map given the camera parameters

    Args:
        depth_map: depth map
        cam_p: camera p matrix
        min_v: amount to crop off the top
        flatten: flatten point cloud to (3, N), otherwise return the point cloud
            in xyz_map (3, H, W) format. (H, W, 3) points can be retrieved using
            xyz_map.transpose(1, 2, 0)
        in_cam0_frame: (optional) If True, shifts the point cloud into cam_0 frame.
            If False, returns the point cloud in the provided camera frame
        use_corr_factors: (optional) If True, applies correction factors along xx and yy
            according to depth in order to reduce projection error.

    Returns:
        point_cloud: (3, N) point cloud
    """

    depth_map_shape = depth_map.shape[0:2]

    if min_v > 0:
        # Crop top part
        depth_map[0:min_v] = 0.0

    xx, yy = np.meshgrid(
        np.linspace(0, depth_map_shape[1] - 1, depth_map_shape[1]),
        np.linspace(0, depth_map_shape[0] - 1, depth_map_shape[0]))

    if use_corr_factors:
        _apply_corr_factor(depth_map, depth_map_shape, xx, yy)

    # Calibration centre x, centre y, focal length
    centre_u = cam_p[0, 2]
    centre_v = cam_p[1, 2]
    focal_length = cam_p[0, 0]

    i = xx - centre_u
    j = yy - centre_v

    # Similar triangles ratio (x/i = d/f)
    ratio = depth_map / focal_length
    x = i * ratio
    y = j * ratio
    z = depth_map

    if in_cam0_frame:
        # Return the points in cam_0 frame
        # Get x offset (b_cam) from calibration: cam_p[0, 3] = (-f_x * b_cam)
        x_offset = -cam_p[0, 3] / focal_length

        # TODO: mask out invalid points
        point_cloud_map = np.asarray([x + x_offset, y, z])

    else:
        # Return the points in the provided camera frame
        point_cloud_map = np.asarray([x, y, z])

    if flatten:
        point_cloud = np.reshape(point_cloud_map, (3, -1))
        return point_cloud.astype(np.float32)
    else:
        return point_cloud_map.astype(np.float32)


def project_depths(point_cloud, cam_p, image_shape, max_depth=100.0):
    """Projects a point cloud into image space and saves depths per pixel.

    Args:
        point_cloud: (3, N) Point cloud in cam0
        cam_p: camera projection matrix
        image_shape: image shape [h, w]
        max_depth: optional, max depth for inversion

    Returns:
        projected_depths: projected depth map
    """

    # Only keep points in front of the camera
    all_points = point_cloud.T

    # Save the depth corresponding to each point
    points_in_img = calib_utils.project_pc_to_image(all_points.T, cam_p)
    points_in_img_int = np.int32(np.round(points_in_img))

    # Remove points outside image
    valid_indices = \
        (points_in_img_int[0] >= 0) & (points_in_img_int[0] < image_shape[1]) & \
        (points_in_img_int[1] >= 0) & (points_in_img_int[1] < image_shape[0])

    all_points = all_points[valid_indices]
    points_in_img_int = points_in_img_int[:, valid_indices]

    # Invert depths
    all_points[:, 2] = max_depth - all_points[:, 2]

    # Only save valid pixels, keep closer points when overlapping
    projected_depths = np.zeros(image_shape)
    valid_indices = [points_in_img_int[1], points_in_img_int[0]]
    projected_depths[valid_indices] = [
        max(projected_depths[
            points_in_img_int[1, idx], points_in_img_int[0, idx]],
            all_points[idx, 2])
        for idx in range(points_in_img_int.shape[1])]

    projected_depths[valid_indices] = \
        max_depth - projected_depths[valid_indices]

    return projected_depths.astype(np.float32)
