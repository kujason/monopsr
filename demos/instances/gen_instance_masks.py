import os
import sys

import cv2
import numpy as np
from monopsr.builders.dataset_builder import DatasetBuilder

from monopsr.core import box_3d_encoder
from monopsr.datasets.kitti import calib_utils
from monopsr.datasets.kitti import obj_utils

INFLATIONS = {
    # Modifications of (x, y, z, l, w, h, ry)
    'Car': np.array([1.0, 1.0, 1.0, 1.25, 1.25, 1.1, 1.0]),
    'Van': np.array([1.0, 1.0, 1.0, 1.1, 1.1, 1.05, 1.0]),
    'Truck': np.array([1.0, 1.0, 1.0, 1.1, 1.1, 1.05, 1.0]),

    'Pedestrian': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.0]),
    'Person_sitting': np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.05, 1.0]),

    'Cyclist': np.array([1.0, 1.0, 1.0, 1.1, 1.1, 1.05, 1.0]),

    'Tram': np.array([1.0, 1.0, 1.0, 1.0, 1.1, 1.05, 1.0]),
    'Misc': np.array([1.0, 1.0, 1.0, 1.05, 1.05, 1.05, 1.0]),
}


def modify_box_3d(box_3d, label):
    """Inflate box_3d to include additional points
    """
    category = label.type

    # Modification of boxes_3d
    inflation = INFLATIONS[category]
    offset = np.array([0.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0])

    modified_box_3d = box_3d * inflation + offset

    return modified_box_3d


def main():

    ##############################
    # Options
    ##############################

    point_cloud_source = 'depth_2_multiscale'

    samples_to_use = None  # all samples

    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAINVAL)

    out_instance_dir = 'outputs/instance_2_{}'.format(point_cloud_source)

    required_classes = [
        'Car',
        'Pedestrian',
        'Cyclist',

        'Van',
        'Truck',
        'Person_sitting',
        'Tram',
        'Misc',
    ]

    ##############################
    # End of Options
    ##############################

    # Create instance folder
    os.makedirs(out_instance_dir, exist_ok=True)

    # Get frame ids to process
    if samples_to_use is None:
        samples_to_use = dataset.get_sample_names()

    # Begin instance mask generation
    for sample_idx, sample_name in enumerate(samples_to_use):

        sys.stdout.write('\r{} / {} Generating {} instances for sample {}'.format(
            sample_idx, dataset.num_samples - 1, point_cloud_source, sample_name))

        # Get image
        image = obj_utils.get_image(sample_name, dataset.image_2_dir)
        image_shape = image.shape[0:2]

        # Get calibration
        frame_calib = calib_utils.get_frame_calib(dataset.calib_dir, sample_name)

        # Get point cloud
        if point_cloud_source.startswith('depth'):
            point_cloud = obj_utils.get_depth_map_point_cloud(
                sample_name, frame_calib, dataset.depth_dir)

        elif point_cloud_source == 'velo':
            point_cloud = obj_utils.get_lidar_point_cloud_for_cam(
                sample_name, frame_calib, dataset.velo_dir, image_shape)
        else:
            raise ValueError('Invalid point cloud source', point_cloud_source)

        # Filter according to classes
        obj_labels = obj_utils.read_labels(dataset.kitti_label_dir, sample_name)
        obj_labels, _ = obj_utils.filter_labels_by_class(obj_labels, required_classes)

        # Get 2D and 3D bounding boxes from labels
        gt_boxes_2d = [box_3d_encoder.object_label_to_box_2d(obj_label)
                       for obj_label in obj_labels]
        gt_boxes_3d = [box_3d_encoder.object_label_to_box_3d(obj_label)
                       for obj_label in obj_labels]

        instance_image = np.full(image_shape, 255, dtype=np.uint8)

        # Start instance index at 0 and generate instance masks for all boxes
        inst_idx = 0
        for obj_label, box_2d, box_3d in zip(obj_labels, gt_boxes_2d, gt_boxes_3d):

            # Apply inflation and offset to box_3d
            modified_box_3d = modify_box_3d(box_3d, obj_label)

            # Get points in 3D box
            box_points, mask = obj_utils.points_in_box_3d(modified_box_3d, point_cloud.T)

            # Get points in 2D box
            points_in_im = calib_utils.project_pc_to_image(box_points.T, cam_p=frame_calib.p2)
            mask_2d = \
                (points_in_im[0] >= box_2d[1]) & \
                (points_in_im[0] <= box_2d[3]) & \
                (points_in_im[1] >= box_2d[0]) & \
                (points_in_im[1] <= box_2d[2])

            if point_cloud_source.startswith('depth'):
                mask_points_in_im = np.where(mask.reshape(image_shape))
                mask_points_in_im = [mask_points_in_im[0][mask_2d], mask_points_in_im[1][mask_2d]]
                instance_pixels = np.asarray([mask_points_in_im[1], mask_points_in_im[0]])
            elif point_cloud_source == 'velo':
                # image_points = box_utils.project_to_image(
                #     box_points.T, frame.p_left).astype(np.int32)
                pass

            # Guarantees that indices don't exceed image dimensions
            instance_pixels[0, :] = np.clip(
                instance_pixels[0, :], 0, image_shape[1] - 1)
            instance_pixels[1, :] = np.clip(
                instance_pixels[1, :], 0, image_shape[0] - 1)

            instance_image[instance_pixels[1, :],
                           instance_pixels[0, :]] = np.uint8(inst_idx)

            inst_idx += 1

        # Write image to directory
        cv2.imwrite(out_instance_dir + '/{}.png'.format(sample_name), instance_image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])


if __name__ == '__main__':
    main()
