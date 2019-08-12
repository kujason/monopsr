import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull
from skimage import filters
import tensorflow as tf

from monopsr.core import evaluation
from monopsr.datasets.kitti import instance_utils, calib_utils
from monopsr.visualization import vis_utils


def np_proj_error(points_uv, points_mask, exp_grid_uv):
    """Calculates projection error of instance points with a 2D box

    Args:
        points_uv: (2, N) Points in u, v coordinates
        points_mask: (N,) Mask of valid points
        exp_grid_uv: expected [u, v] grid projection

    Returns:
        proj_err_norm: projection error normalized by the number of valid pixels
    """

    # Calculation projection error
    pred_grid_uv = points_uv.reshape(2, *exp_grid_uv[0].shape)
    points_mask = points_mask.reshape(1, *exp_grid_uv[0].shape)
    pred_proj_err_uv = pred_grid_uv - exp_grid_uv
    pred_proj_err = np.sum(np.abs(pred_proj_err_uv) * points_mask)
    proj_err_norm = pred_proj_err / np.count_nonzero(points_mask)

    return proj_err_norm


def scipy_proj_error(x, args):
    """Calculates projection error of instance points with a 2D box.
    Used for minimizing projection error when varying xz_dist and centroid_y.

    Args:
        x: array of inputs
            xz_dist: distance along viewing angle
            centroid_y: box centroid y
        args: dict with additional data
            'viewing_angle': viewing angle
            'inst_points' = (N, 3) instance points
            'cam_p' = (3, 4) camera projection matrix
            'exp_grid_uv' = expected [u, v] grid projection
            'rotate_view' = bool of whether to rotate by viewing angle

    Returns:
        proj_err_norm: projection error normalized by the number of valid pixels
    """

    # Parse inputs from x
    xz_dist = x[0]
    centroid_y = x[1]

    # Parse inputs from args
    viewing_angle = args['viewing_angle']
    inst_points = args['inst_points']
    cam_p = args['cam_p']
    exp_grid_uv = args['exp_grid_uv']
    rotate_view = args['rotate_view']

    pred_points_in_img, valid_points_mask = instance_utils.proj_points(
        xz_dist, centroid_y, viewing_angle, inst_points, cam_p, rotate_view=rotate_view)
    proj_err_norm = np_proj_error(pred_points_in_img, valid_points_mask, exp_grid_uv)

    return proj_err_norm


def scipy_proj_error_with_viewing_angle(x, args):
    """Calculates projection error of instance points with a 2D box.
    Used for minimizing projection error when varying xz_dist, centroid_y, and viewing_angle.

    Args:
        x: array of inputs
            xz_dist: distance along viewing angle
            centroid_y: box centroid y
            viewing_angle: viewing angle

        args: dict with additional data
            'inst_points' = (N, 3) instance points
            'cam_p' = (3, 4) camera projection matrix
            'exp_grid_uv' = expected [u, v] grid projection
            'rotate_view' = bool of whether to rotate by viewing angle

    Returns:
        proj_err_norm: projection error normalized by the number of valid pixels
    """

    # Parse inputs from x
    xz_dist = x[0]
    centroid_y = x[1]
    viewing_angle = x[2]

    # Parse inputs from args
    inst_points = args['inst_points']
    cam_p = args['cam_p']
    exp_grid_uv = args['exp_grid_uv']
    rotate_view = args['rotate_view']

    pred_points_in_img, valid_points_mask = instance_utils.proj_points(
        xz_dist, centroid_y, viewing_angle, inst_points, cam_p, rotate_view=rotate_view)
    proj_err_norm = np_proj_error(pred_points_in_img, valid_points_mask, exp_grid_uv)

    return proj_err_norm


def tf_proj_error(points_uv, points_mask, exp_grid_uv):
    """

    Args:
        points_uv:
        points_mask:
        exp_grid_uv:

    Returns:

    """

    # return tf.zeros(32)
    raise NotImplementedError('Not implemented yet')


def np_proj_err_rgb_images(xz_dist, centroid_y, viewing_angle,
                           cam2_inst_points_local, cam_p,
                           inst_rgb, inst_mask, image, valid_mask_map, box_2d,
                           guess_row_col, show_images=False):
    """(Work in progress) Calculates the projection error based on RGB similarity and shows
    images for comparison.

    Args:
        xz_dist: Distance along viewing angle
        centroid_y: Object centroid y
        viewing_angle: Viewing angle
        cam2_inst_points_local: (N, 3) Instance points in local frame
        cam_p: (3, 4) Camera projection matrix
        inst_rgb: List of instance RGB values
        image: Image of sample
        valid_mask_map: (H, W) Map mask of valid values
        guess_row_col: Guess index, used for numbering images
        show_images: (optional) Whether to show comparison images

    Returns:
        image_diff_total: Lowest image difference
    """

    # Get projection into image
    proj_uv, valid_points_mask = instance_utils.proj_points(
        xz_dist, centroid_y, viewing_angle, cam2_inst_points_local, cam_p)

    # Get RGB values of projected pixels
    proj_uv_int = np.round(proj_uv).astype(np.int32)
    guess_rgb = image[proj_uv_int[1], proj_uv_int[0]]
    guess_rgb_map = guess_rgb.reshape(48, 48, 3) * np.expand_dims(valid_mask_map, 2)

    # Estimated image
    est_image = np.copy(image) * np.expand_dims(~inst_mask, 2)
    est_image[proj_uv_int[1], proj_uv_int[0]] = inst_rgb

    est_image[proj_uv_int[1]-1, proj_uv_int[0]] = inst_rgb
    est_image[proj_uv_int[1]+1, proj_uv_int[0]] = inst_rgb
    est_image[proj_uv_int[1], proj_uv_int[0]-1] = inst_rgb
    est_image[proj_uv_int[1], proj_uv_int[0]+1] = inst_rgb

    box_2d_int = np.round(box_2d).astype(np.int32)
    est_inst_rgb = est_image[box_2d_int[0]:box_2d_int[2], box_2d_int[1]:box_2d_int[3]]
    est_inst_rgb_resized = cv2.resize(est_inst_rgb, (48, 48))

    # Check image similarity
    inst_rgb_map = inst_rgb.reshape(48, 48, 3)
    # image_diff_map = abs(inst_rgb_map - guess_rgb_map)
    image_diff_map = abs(inst_rgb_map - est_inst_rgb_resized)
    image_diff_map_norm = np.sum(image_diff_map, axis=2) / 255.0
    image_diff_total = np.sum(image_diff_map_norm)

    if show_images:
        # cv2_size = (160, 160)
        cv2_size = (90, 90)
        cv2_size = (120, 120)

        # # Show instance RGB for comparison
        # inst_rgb_map_resized = cv2.resize(inst_rgb_map, cv2_size)
        # vis_utils.cv2_imshow('inst_rgb_map_resized {}'.format(guess_row_col),
        #                      inst_rgb_map_resized,
        #                      size_wh=cv2_size, row_col=guess_row_col)
        #
        # # Show guess
        # guess_rgb_map_resized = cv2.resize(guess_rgb_map, (200, 200))
        # vis_utils.cv2_imshow('guess_rgb_map_resized {}'.format(guess_row_col),
        #                      guess_rgb_map_resized,
        #                      size_wh=cv2_size, row_col=guess_row_col)


        vis_utils.cv2_imshow('est_inst_rgb_resized {}'.format(guess_row_col),
                             est_inst_rgb_resized,
                             size_wh=cv2_size, row_col=guess_row_col)

        # combined = cv2.addWeighted(inst_rgb_map, 0.5, est_inst_rgb_resized, 0.5, 0.0)
        # vis_utils.cv2_imshow('combined {}'.format(guess_row_col),
        #                      combined,
        #                      size_wh=cv2_size, row_col=guess_row_col)

        # vis_utils.cv2_imshow('image_diff_map_norm {}'.format(guess_row_col),
        #                      image_diff_map_norm,
        #                      size_wh=cv2_size, row_col=guess_row_col)

        # vis_utils.cv2_imshow('valid_mask {}'.format(centroid_y),
        #                      (valid_mask_map * 255).astype(np.uint8),
        #                      size_wh=cv2_size, row_col=guess_row_col)

    return image_diff_total


def np_proj_err_rgb(xz_dist, centroid_y, viewing_angle, cam2_inst_points_local, cam_p,
                    inst_rgb, image, valid_mask_map):

    # Get instance RGB
    inst_rgb_map = inst_rgb.reshape(48, 48, 3)

    # Project points to image
    proj_uv, _ = instance_utils.proj_points(
        xz_dist, centroid_y, viewing_angle, cam2_inst_points_local, cam_p)

    # Get RGB values of projected pixels
    proj_uv_int = np.round(proj_uv).astype(np.int32)
    guess_rgb = image[proj_uv_int[1], proj_uv_int[0]]
    guess_rgb_map = guess_rgb.reshape(48, 48, 3) * np.expand_dims(valid_mask_map, 2)

    # Check image similarity
    image_diff_map = abs(inst_rgb_map - guess_rgb_map)
    image_diff_map_norm = np.sum(image_diff_map, axis=2) / 255.0
    image_diff_total = np.sum(image_diff_map_norm) / np.count_nonzero(valid_mask_map)

    return image_diff_total


def scipy_proj_err_rgb(x, args):
    """Calculates projection error based on RGB similarity.
    (Minimization with this doesn't seem to work since
    large patches will be matched at incorrect positions)
    """

    # Parse inputs from x
    xz_dist = x[0]
    centroid_y = x[1]

    if len(x) == 3:
        viewing_angle = x[2]
    else:
        viewing_angle = args['viewing_angle']

    # Parse inputs from args
    inst_points = args['inst_points']
    cam_p = args['cam_p']

    inst_rgb = args['inst_rgb']
    image = args['image']
    valid_mask_map = args['valid_mask_map']

    proj_err_rgb = np_proj_err_rgb(
        xz_dist=xz_dist,
        centroid_y=centroid_y,
        viewing_angle=viewing_angle,
        cam2_inst_points_local=inst_points,
        cam_p=cam_p,
        inst_rgb=inst_rgb,
        image=image,
        valid_mask_map=valid_mask_map,
    )

    return proj_err_rgb


def convex_hull_mask_iou(points_uv, im_shape, gt_hull_mask):
    """Computes masks by calculating a convex hull from points. Creates two masks (if possible),
    one for the estimated foreground pixels and one for the estimated background pixels.

    Args:
        points_uv: (2, N) Points in u, v coordinates
        im_shape: image shape [image_height, im_width]
        gt_hull_mask: mask created by calculating convex hull

    Returns:
        best_iou: best mask iou calculated from the calculated hull masks and the ground truth hull
        mask
    """

    im_height, im_width = im_shape

    # Segment the points into background and foreground
    if len(set(points_uv[0])) > 1:
        thresh = filters.threshold_li(points_uv[0])
        pred_seg_1 = points_uv[0] > thresh
        pred_seg_2 = points_uv[0] < thresh
        segs = [pred_seg_1, pred_seg_2]
    else:
        # There is only one unique point so a threshold cannot be made
        segs = [np.full(points_uv[0].shape, True, dtype=bool)]

    mask_list = []
    # Loop over both segments since it is uncertain which segment is foreground or background
    for seg in segs:

        # Obtain the coordinates of the pixels
        pred_u = np.int32(points_uv[0][seg])
        pred_v = np.int32(points_uv[1][seg])
        # Remove duplicate coordinates by forming a set
        coords = set(zip(pred_u, pred_v))
        # Convex hull calculation requires a numpy array
        coords = np.array(list(coords))

        # Need at least 3 points to create convex hull
        if len(coords) < 3:
            continue
        # Points must not lie along a single line in order to create convex hull
        elif any(np.all(coords == coords[0, :], axis=0)):
            continue
        else:
            hull = ConvexHull(coords)

            img = Image.new('L', (im_width, im_height), 0)
            vertices = list(zip(coords[hull.vertices, 0], coords[hull.vertices, 1]))
            ImageDraw.Draw(img).polygon(vertices, outline=1, fill=1)
            mask = np.array(img)
            mask_list.append(mask)

    best_iou = 0
    for mask in mask_list:
        iou = evaluation.mask_iou(mask, gt_hull_mask)
        if iou > best_iou:
            best_iou = iou

    return best_iou


def scipy_convex_hull_mask_inv_iou(x, args):
    """Computes masks by calculating a convex hull from points. Creates two masks (if possible),
    one for the estimated foreground pixels and one for the estimated background pixels.
    Minimizes inverted IoU by varying xz_dist and centroid_y.

    Args:
        x: array of inputs
            xz_dist: distance along viewing angle
            centroid_y: box centroid y

        args: dict with additional data
            'viewing_angle': viewing angle
            'inst_points' = (N, 3) instance points
            'cam_p' = (3, 4) camera projection matrix
            'im_shape' = image shape [im_height, im_width]
            'gt_hull_mask' = expected mask created from instance mask

    Returns:
        inverted_iou: 1.0 - IoU of the mask computed from the convex hull and the gt hull mask
    """

    # Parse inputs from x
    xz_dist = x[0]
    centroid_y = x[1]

    # Parse inputs from args
    viewing_angle = args['viewing_angle']
    inst_points = args['inst_points']
    cam_p = args['cam_p']
    im_shape = args['im_shape']
    gt_hull_mask = args['gt_hull_mask']

    pred_points_in_img, valid_points_mask = instance_utils.proj_points(
        xz_dist, centroid_y, viewing_angle, inst_points, cam_p)
    iou = convex_hull_mask_iou(pred_points_in_img, im_shape, gt_hull_mask)

    # Invert IoU so it can be minimized
    inverted_iou = 1.0 - iou

    return inverted_iou


def scipy_convex_hull_mask_inv_iou_with_viewing_angle(x, args):
    """Computes masks by calculating a convex hull from points. Creates two masks (if possible),
    one for the estimated foreground pixels and one for the estimated background pixels.
    Minimizes inverted IoU by varying xz_dist, centroid_y, and viewing angle.

    Args:
        x: array of inputs
            xz_dist: distance along viewing angle
            centroid_y: box centroid y
            viewing_angle: viewing angle

        args: dict with additional data
            'viewing_angle': viewing angle
            'inst_points' = (N, 3) instance points
            'cam_p' = (3, 4) camera projection matrix
            'im_shape' = image shape [im_height, im_width]
            'gt_hull_mask' = expected mask created from instance mask

    Returns:
        inverted_iou: 1.0 - IoU of the mask computed from the convex hull and the gt hull mask
    """

    # Parse inputs from x
    xz_dist = x[0]
    centroid_y = x[1]
    viewing_angle = x[2]

    # Parse inputs from args
    inst_points = args['inst_points']
    cam_p = args['cam_p']
    im_shape = args['im_shape']
    gt_hull_mask = args['gt_hull_mask']

    pred_points_in_img, valid_points_mask = instance_utils.proj_points(
        xz_dist, centroid_y, viewing_angle, inst_points, cam_p)
    iou = convex_hull_mask_iou(pred_points_in_img, im_shape, gt_hull_mask)

    # Invert IoU so it can be minimized
    inverted_iou = 1.0 - iou

    return inverted_iou
