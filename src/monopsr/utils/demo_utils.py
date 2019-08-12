import copy

import numpy as np
import tensorflow as tf

import monopsr
from monopsr.core import config_utils
from monopsr.core import box_3d_encoder
from monopsr.core import evaluation
from monopsr.datasets.kitti import obj_utils
from monopsr.datasets.kitti.obj_utils import Difficulty
from monopsr.visualization import vis_utils

COLOUR_SCHEME_PREDICTIONS = {
    "Easy GT": (255, 255, 0),     # Yellow
    "Medium GT": (255, 128, 0),   # Orange
    "Hard GT": (255, 0, 0),       # Red

    "Prediction": (50, 255, 50),  # Green
}


class Checkpoint:
    def __init__(self, checkpoint_name, global_step):
        self.name = checkpoint_name
        self.step = global_step


def get_point_cloud(pc_source, sample_name, frame_calib,
                    velo_dir=None, depth_dir=None, disp_dir=None,
                    image_shape=None, cam_idx=2):

    if pc_source == 'lidar':
        point_cloud = obj_utils.get_lidar_point_cloud_for_cam(
            sample_name, frame_calib, velo_dir, image_shape, cam_idx)
    elif pc_source == 'depth':
        point_cloud = obj_utils.get_depth_map_point_cloud(
            sample_name, frame_calib, depth_dir)
    elif pc_source == 'stereo':
        raise NotImplementedError('Not implemented yet')
    else:
        raise ValueError('Invalid point cloud source', pc_source)

    return point_cloud


def get_gts_based_on_difficulty(dataset, sample_name):
    """Returns lists of ground-truth based on difficulty.
    """
    # Get all ground truth labels and filter to dataset classes
    all_gt_objs = obj_utils.read_labels(dataset.kitti_label_dir, sample_name)
    gt_objs, _ = obj_utils.filter_labels_by_class(all_gt_objs, dataset.classes)

    # Filter objects to desired difficulty
    easy_gt_objs, _ = obj_utils.filter_labels_by_difficulty(
        copy.deepcopy(gt_objs), difficulty=Difficulty.EASY)
    medium_gt_objs, _ = obj_utils.filter_labels_by_difficulty(
        copy.deepcopy(gt_objs), difficulty=Difficulty.MODERATE)
    hard_gt_objs, _ = obj_utils.filter_labels_by_difficulty(
        copy.deepcopy(gt_objs), difficulty=Difficulty.HARD)

    for gt_obj in easy_gt_objs:
        gt_obj.type = 'Easy GT'
    for gt_obj in medium_gt_objs:
        gt_obj.type = 'Medium GT'
    for gt_obj in hard_gt_objs:
        gt_obj.type = 'Hard GT'

    return easy_gt_objs, medium_gt_objs, hard_gt_objs, all_gt_objs


def get_filtered_pc_and_colours(dataset,
                                image,
                                img_idx):

    image_size = [image.shape[1], image.shape[0]]
    point_cloud = obj_utils.get_depth_map_point_cloud(img_idx,
                                                      dataset.calib_dir,
                                                      dataset.depth_dir,
                                                      im_size=image_size)
    point_cloud = np.asarray(point_cloud)

    # Filter point cloud to extents
    area_extents = np.asarray([[-40, 40], [-5, 3], [0, 70]])

    points = point_cloud.T
    point_filter = obj_utils.get_ground_offset_filter(point_cloud, area_extents)
    points = points[point_filter]

    point_colours = vis_utils.project_img_to_point_cloud(points,
                                                         image,
                                                         dataset.calib_dir,
                                                         img_idx)

    return points, point_colours


def get_max_ious_3d(all_gt_boxes_3d, pred_boxes_3d):
    """Helper function to calculate 3D IoU for the given predictions.

    Args:
        all_gt_boxes_3d: A list of the same ground-truth boxes in box_3d
            format.
        pred_boxes_3d: A list of predictions in box_3d format.
    """

    # Only calculate ious if there are predictions
    if pred_boxes_3d:
        # Convert to iou format
        gt_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
            all_gt_boxes_3d)
        pred_objs_iou_fmt = box_3d_encoder.box_3d_to_3d_iou_format(
            pred_boxes_3d)

        max_ious_3d = np.zeros(len(all_gt_boxes_3d))
        for gt_obj_idx in range(len(all_gt_boxes_3d)):

            gt_obj_iou_fmt = gt_objs_iou_fmt[gt_obj_idx]

            ious_3d = evaluation.three_d_iou(gt_obj_iou_fmt,
                                             pred_objs_iou_fmt)

            max_ious_3d[gt_obj_idx] = np.amax(ious_3d)
    else:
        # No detections, all ious = 0
        max_ious_3d = np.zeros(len(all_gt_boxes_3d))

    return max_ious_3d


def create_demo_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess


def get_experiment_info(checkpoint_name):
    exp_output_base_dir = monopsr.data_dir() + '/outputs/' + checkpoint_name

    # Parse experiment config
    config_file = exp_output_base_dir + '/{}.yaml'.format(checkpoint_name)
    config = config_utils.parse_yaml_config(config_file)

    predictions_base_dir = exp_output_base_dir + '/predictions'

    return config, predictions_base_dir
