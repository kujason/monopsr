import fnmatch
import os

import monopsr
import numpy as np
import cv2

from monopsr.core import orientation_encoder
from monopsr.core import constants
from monopsr.datasets.kitti import calib_utils
from monopsr.datasets.kitti import instance_utils
from monopsr.datasets.kitti import kitti_aug
from monopsr.datasets.kitti import obj_utils
from monopsr.datasets.kitti.obj_utils import Difficulty


class Sample:
    def __init__(self, name, augs):
        self.name = name
        self.augs = augs

    def __repr__(self):
        return '({}, augs: {})'.format(self.name, self.augs)


class KittiDataset:

    def __init__(self, dataset_config, train_val_test):

        self.dataset_config = dataset_config
        self.train_val_test = train_val_test

        # Parse config
        self.name = self.dataset_config.name

        self.data_split = self.dataset_config.data_split
        self.dataset_dir = os.path.expanduser(self.dataset_config.dataset_dir)
        data_split_dir = self.dataset_config.data_split_dir

        self.num_boxes = self.dataset_config.num_boxes
        self.num_alpha_bins = self.dataset_config.num_alpha_bins
        self.alpha_bin_overlap = self.dataset_config.alpha_bin_overlap
        self.centroid_type = self.dataset_config.centroid_type

        # TODO: set up both cameras if possible
        self.cam_idx = 2

        self.classes = list(self.dataset_config.classes)
        self.num_classes = len(self.classes)

        # Object filtering config
        if self.train_val_test in ['train', 'val']:
            obj_filter_config = self.dataset_config.obj_filter_config
            obj_filter_config.classes = self.classes
            self.obj_filter = obj_utils.ObjectFilter(obj_filter_config)

        else:  # self.train_val_test == 'test'
            # Use all detections during inference
            self.obj_filter = obj_utils.ObjectFilter.create_obj_filter(
                classes=self.classes,
                difficulty=Difficulty.ALL,
                occlusion=None,
                truncation=None,
                box_2d_height=None,
                depth_range=None)

        self.has_kitti_labels = self.dataset_config.has_kitti_labels

        # MSCNN settings
        self.use_mscnn_detections = self.dataset_config.use_mscnn_detections
        self.mscnn_thr = self.dataset_config.mscnn_thr

        # Always use statistics computed using KITTI 2D boxes
        self.trend_data = 'kitti'

        self.classes_name = self._set_up_classes_name()

        if self.classes_name == 'Car':
            self.mscnn_merge_min_iou = 0.7
        elif self.classes_name in ['Pedestrian', 'Cyclist']:
            self.mscnn_merge_min_iou = 0.5

        # Check that paths and split are valid
        self._check_dataset_dir()
        all_dataset_files = os.listdir(self.dataset_dir)
        self._check_data_split_valid(all_dataset_files)
        self.data_split_dir = self._check_data_split_dir_valid(
            all_dataset_files, data_split_dir)

        self.depth_version = self.dataset_config.depth_version
        self.instance_version = self.dataset_config.instance_version

        # Setup directories
        self._set_up_directories()

        # Whether to oversample objects to required number of boxes
        self.oversample = self.dataset_config.oversample

        # Augmentation
        self.aug_config = self.dataset_config.aug_config

        # Initialize the sample list
        loaded_sample_names = self.load_sample_names(self.data_split)
        all_samples = [Sample(sample_name, []) for sample_name in loaded_sample_names]

        self.sample_list = np.asarray(all_samples)
        self.num_samples = len(self.sample_list)

        # Load cluster info
        # TODO: get clusters properly
        self.clusters = [3.892, 1.619, 1.530]
        self.std_devs = [0.440, 0.106, 0.138]

        # Batch pointers
        self._index_in_epoch = 0
        self.epochs_completed = 0

    def _check_dataset_dir(self):
        """Checks that dataset directory exists in the file system

        Raises:
            FileNotFoundError: if the dataset folder is missing
        """
        # Check that dataset path is valid
        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError('Dataset path does not exist: {}'
                                    .format(self.dataset_dir))

    def _check_data_split_valid(self, all_dataset_files):
        possible_splits = []
        for file_name in all_dataset_files:
            if fnmatch.fnmatch(file_name, '*.txt'):
                possible_splits.append(os.path.splitext(file_name)[0])
        # This directory contains a readme.txt file, remove it from the list
        if 'readme' in possible_splits:
            possible_splits.remove('readme')

        if self.data_split not in possible_splits:
            raise ValueError("Invalid data split: {}, possible_splits: {}"
                             .format(self.data_split, possible_splits))

    def _check_data_split_dir_valid(self, all_dataset_files, data_split_dir):
        # Check data_split_dir
        # Get possible data split dirs from folder names in dataset folder
        possible_split_dirs = []
        for folder_name in all_dataset_files:
            if os.path.isdir(self.dataset_dir + '/' + folder_name):
                possible_split_dirs.append(folder_name)

        if data_split_dir in possible_split_dirs:
            # Overwrite with full path
            data_split_dir = self.dataset_dir + '/' + data_split_dir
        else:
            raise ValueError(
                "Invalid data split dir: {}, possible dirs".format(
                    data_split_dir, possible_split_dirs))

        return data_split_dir

    def _set_up_directories(self):
        """Sets up data directories."""
        # Setup Directories
        self.rgb_image_dir = self.data_split_dir + '/image_' + str(self.cam_idx)
        self.image_2_dir = self.data_split_dir + '/image_2'
        self.image_3_dir = self.data_split_dir + '/image_3'

        self.calib_dir = self.data_split_dir + '/calib'
        self.disp_dir = self.data_split_dir + '/disparity'
        self.planes_dir = self.data_split_dir + '/planes'
        self.velo_dir = self.data_split_dir + '/velodyne'
        self.depth_dir = self.data_split_dir + '/depth_{}_{}'.format(
            self.cam_idx, self.depth_version)
        self.instance_dir = self.data_split_dir + '/instance_{}_{}'.format(
            self.cam_idx, self.instance_version)

        self.mscnn_label_dir = monopsr.data_dir() + \
            '/detections/mscnn/kitti_fmt/{}/merged_{}/data'.format(
                self.data_split, '_'.join(map(str, self.mscnn_thr)))

        if self.has_kitti_labels:
            self.kitti_label_dir = self.data_split_dir + '/label_2'

    def _set_up_classes_name(self):
        # Unique identifier for multiple classes
        if self.num_classes > 1:
            raise NotImplementedError('Number of classes must be 1')
        else:
            classes_name = self.classes[0]

        return classes_name

    def get_sample_names(self):
        return [sample.name for sample in self.sample_list]

    # Get sample paths
    def get_rgb_image_path(self, sample_name):
        return self.rgb_image_dir + '/' + sample_name + '.png'

    def get_image_2_path(self, sample_name):
        return self.image_2_dir + '/' + sample_name + '.png'

    def get_image_3_path(self, sample_name):
        return self.image_3_dir + '/' + sample_name + '.png'

    def get_depth_map_path(self, sample_name):
        return self.depth_dir + '/' + sample_name + '_left_depth.png'

    def get_velodyne_path(self, sample_name):
        return self.velo_dir + '/' + sample_name + '.bin'

    # Cluster info
    def get_cluster_info(self):
        return self.clusters, self.std_devs

    # Data loading methods
    def load_sample_names(self, data_split):
        """Load the sample names listed in this dataset's set file
        (e.g. train.txt, validation.txt)

        Args:
            data_split: the sample list to load

        Returns:
            A list of sample names (file names) read from
            the .txt file corresponding to the data split
        """
        set_file = self.dataset_dir + '/' + data_split + '.txt'
        with open(set_file, 'r') as f:
            sample_names = f.read().splitlines()

        return np.asarray(sample_names)

    def get_sample_dict(self, indices):
        """ Loads input-output data for a set of samples. Should only be
            called when a particular sample dict is required. Otherwise,
            samples should be provided by the next_batch function

        Args:
            indices: A list of sample indices from the dataset.sample_list
                to be loaded

        Return:
            samples: a list of data sample dicts
        """
        sample_dicts = []
        for sample_idx in indices:

            sample = self.sample_list[sample_idx]
            sample_name = sample.name

            # Load image (BGR -> RGB)
            bgr_image = cv2.imread(self.get_rgb_image_path(sample_name))
            rgb_image = bgr_image[..., :: -1]
            image_shape = rgb_image.shape[0:2]
            image_input = rgb_image

            # Get calibration
            frame_calib = calib_utils.get_frame_calib(self.calib_dir, sample_name)
            cam_p = frame_calib.p2

            # Only read labels if they exist
            if self.train_val_test in ['train', 'val']:

                # Read KITTI object labels
                kitti_obj_labels = obj_utils.read_labels(self.kitti_label_dir, sample_name)

                if self.use_mscnn_detections and self.train_val_test == 'val':
                    # Read mscnn obj labels and replace the KITTI obj label box coords and scores
                    mscnn_obj_labels = obj_utils.read_labels(self.mscnn_label_dir, sample_name)

                    obj_labels = obj_utils.merge_kitti_and_mscnn_obj_labels(
                        kitti_obj_labels, mscnn_obj_labels, min_iou=self.mscnn_merge_min_iou,
                        default_score_type='distance')
                else:
                    obj_labels = kitti_obj_labels

                num_all_objs = len(obj_labels)

                # Filter labels
                obj_labels, obj_mask = obj_utils.apply_obj_filter(obj_labels,
                                                                  self.obj_filter)
                num_objs = len(obj_labels)
                if num_objs < 1:
                    sample_dicts.append(None)
                    continue

                if self.use_mscnn_detections:
                    # Get filtered original kitti_obj_labels
                    kitti_obj_labels, kitti_obj_mask = obj_utils.apply_obj_filter(kitti_obj_labels,
                                                                                  self.obj_filter)
                    num_kitti_objs = len(kitti_obj_labels)
                    if num_kitti_objs < 1:
                        sample_dicts.append(None)
                        continue

                # Load instance masks
                instance_image = instance_utils.get_instance_image(sample_name, self.instance_dir)
                instance_masks = instance_utils.get_instance_mask_list(instance_image, num_all_objs)
                instance_masks = instance_masks[obj_mask]

                if self.oversample:
                    # Oversample to required number of boxes
                    num_to_oversample = self.num_boxes - num_objs

                    oversample_indices = np.random.choice(num_objs, num_to_oversample, replace=True)
                    oversample_indices = np.hstack([np.arange(0, num_objs), oversample_indices])
                    obj_labels = obj_labels[oversample_indices]
                    instance_masks = instance_masks[oversample_indices]

                # Augmentation if in train mode
                if self.train_val_test == 'train':

                    # Image augmentation
                    use_image_aug = self.aug_config.use_image_aug
                    if use_image_aug:
                        image_input = kitti_aug.apply_image_noise(rgb_image)

                    # Box jittering
                    box_jitter_type = self.aug_config.box_jitter_type
                    if box_jitter_type is None:
                        pass
                    elif box_jitter_type == 'oversample':
                        # Replace oversampled boxes with jittered boxes
                        if not self.oversample:
                            raise ValueError('Must oversample object labels to use {} '
                                             'box jitter type'.format(box_jitter_type))
                        aug_labels = kitti_aug.jitter_obj_boxes_2d(
                            obj_labels[num_objs:], 0.7, image_shape)
                        obj_labels[num_objs:] = aug_labels
                    elif box_jitter_type == 'oversample_gt':
                        # Replace oversampled boxes with jittered gt boxes
                        if not self.oversample:
                            raise ValueError('Must oversample object labels to use {} '
                                             'box jitter type'.format(box_jitter_type))

                        # Get enough gt boxes to jitter
                        gt_num_to_oversample = self.num_boxes - num_objs
                        gt_oversample_indices = np.random.choice(num_kitti_objs,
                                                                 gt_num_to_oversample,
                                                                 replace=True)
                        kitti_obj_labels = kitti_obj_labels[gt_oversample_indices]

                        aug_labels = kitti_aug.jitter_obj_boxes_2d(
                            kitti_obj_labels, 0.7, image_shape)
                        obj_labels[num_objs:] = aug_labels
                    elif box_jitter_type == 'all':
                        # Apply data augmentation on all labels
                        obj_labels = kitti_aug.jitter_obj_boxes_2d(
                            obj_labels, 0.7, image_shape)
                    else:
                        raise ValueError('Invalid box_jitter_type', box_jitter_type)

                # TODO: Do this some other way
                # Get 2D and 3D boxes
                label_boxes_2d = obj_utils.boxes_2d_from_obj_labels(obj_labels)
                label_boxes_3d = obj_utils.boxes_3d_from_obj_labels(obj_labels)
                label_alphas = np.asarray(
                    [obj_label.alpha for obj_label in obj_labels], dtype=np.float32)

                label_alpha_bins, label_alpha_regs, label_valid_alpha_bins = \
                    zip(*[orientation_encoder.np_orientation_to_angle_bin(
                        obj_label.alpha, self.num_alpha_bins, self.alpha_bin_overlap)
                        for obj_label in obj_labels])

                # Get viewing angles
                label_viewing_angles_2d = np.asarray(
                    [obj_utils.get_viewing_angle_box_2d(box_2d, cam_p)
                     for box_2d in label_boxes_2d], dtype=np.float32)
                label_viewing_angles_3d = np.asarray(
                    [obj_utils.get_viewing_angle_box_3d(box_3d, cam_p)
                     for box_3d in label_boxes_3d], dtype=np.float32)

                # Parse class indices
                label_class_indices = [
                    obj_utils.class_str_to_index(obj_label.type, self.classes)
                    for obj_label in obj_labels]
                label_class_indices = np.expand_dims(np.asarray(label_class_indices,
                                                                dtype=np.int32), axis=1)
                label_class_strs = [obj_label.type for obj_label in obj_labels]

                # Get proposal z centroid offset
                prop_cen_z_offset_list = np.asarray([instance_utils.get_prop_cen_z_offset(
                    class_str) for class_str in label_class_strs])

                # Get xyz map in cam_N frame
                depth_map = obj_utils.get_depth_map(sample_name, self.depth_dir)

                # Get scores
                label_scores = np.asarray([obj_label.score
                                           for obj_label in obj_labels], np.float32)

                # Get lwh average
                lwh_means = np.asarray([obj_utils.get_mean_lwh_and_std_dev(class_str)[0]
                                       for class_str in label_class_strs])

            elif self.train_val_test == 'test':
                # Read object test labels
                obj_labels = obj_utils.read_labels(self.mscnn_label_dir, sample_name)
                num_objs = len(obj_labels)
                if num_objs < 1:
                    sample_dicts.append(None)
                    continue

                # Just filter classes
                obj_labels, obj_mask = obj_utils.apply_obj_filter(obj_labels,
                                                                  self.obj_filter)
                num_objs = len(obj_labels)
                if num_objs < 1:
                    sample_dicts.append(None)
                    continue

                # Oversample to required number of boxes
                num_to_oversample = self.num_boxes - num_objs
                oversample_indices = np.random.choice(num_objs, num_to_oversample, replace=True)
                oversample_indices = np.hstack([np.arange(0, num_objs), oversample_indices])
                obj_labels = obj_labels[oversample_indices]

                # Get 2D boxes
                label_boxes_2d = obj_utils.boxes_2d_from_obj_labels(obj_labels)

                # Get score
                label_scores = np.asarray([obj_label.score
                                           for obj_label in obj_labels], np.float32)

                # Calculate viewing angles
                label_viewing_angles_2d = np.asarray(
                    [obj_utils.get_viewing_angle_box_2d(box_2d, cam_p)
                     for box_2d in label_boxes_2d], dtype=np.float32)

                label_class_indices = [
                    obj_utils.class_str_to_index(obj_label.type, self.classes)
                    for obj_label in obj_labels]
                label_class_indices = np.expand_dims(np.asarray(label_class_indices,
                                                                dtype=np.int32), axis=1)
                label_class_strs = [obj_label.type for obj_label in obj_labels]

                # Get lwh average
                lwh_means = np.asarray([obj_utils.get_mean_lwh_and_std_dev(class_str)[0]
                                       for class_str in label_class_strs])

                # Get proposal z centroid offset
                prop_cen_z_offset_list = np.asarray([instance_utils.get_prop_cen_z_offset(
                    class_str) for class_str in label_class_strs])

            else:
                raise ValueError('Invalid run mode', self.train_val_test)

            # Common inputs for all train_val_test modes
            # Normalize 2D boxes
            label_boxes_2d_norm = label_boxes_2d / np.tile(image_shape, 2)

            sample_dict = {
                constants.SAMPLE_NUM_OBJS: num_objs,

                constants.SAMPLE_IMAGE_INPUT: image_input,

                constants.SAMPLE_CAM_P: cam_p,
                constants.SAMPLE_NAME: sample_name,

                constants.SAMPLE_LABEL_BOXES_2D_NORM: label_boxes_2d_norm,
                constants.SAMPLE_LABEL_BOXES_2D: label_boxes_2d,
                constants.SAMPLE_LABEL_SCORES: label_scores,
                constants.SAMPLE_LABEL_CLASS_STRS: np.expand_dims(label_class_strs, 1),
                constants.SAMPLE_LABEL_CLASS_INDICES: label_class_indices,

                constants.SAMPLE_MEAN_LWH: lwh_means,

                constants.SAMPLE_PROP_CEN_Z_OFFSET: prop_cen_z_offset_list,

                constants.SAMPLE_VIEWING_ANGLES_2D: label_viewing_angles_2d,
            }

            if self.train_val_test in ['train', 'val']:

                sample_dict.update({

                    constants.SAMPLE_LABEL_BOXES_3D: label_boxes_3d,
                    constants.SAMPLE_ALPHAS: label_alphas,
                    constants.SAMPLE_ALPHA_BINS: np.asarray(label_alpha_bins),
                    constants.SAMPLE_ALPHA_REGS: np.asarray(label_alpha_regs),
                    constants.SAMPLE_ALPHA_VALID_BINS: np.asarray(label_valid_alpha_bins),

                    constants.SAMPLE_VIEWING_ANGLES_3D: label_viewing_angles_3d,

                    constants.SAMPLE_INSTANCE_MASKS: instance_masks,

                    constants.SAMPLE_DEPTH_MAP: depth_map,

                })

            elif self.train_val_test == 'test':
                # No additional labels for test mode
                pass

            sample_dicts.append(sample_dict)

        return sample_dicts

    def _shuffle_samples(self):
        perm = np.arange(self.num_samples)
        np.random.shuffle(perm)
        self.sample_list = self.sample_list[perm]

    def next_batch(self, batch_size, shuffle):
        """
        Retrieve the next `batch_size` samples from this data set.

        Args:
            batch_size: number of samples in the batch
            shuffle: whether to shuffle the indices after an epoch is completed

        Returns:
            list of dictionaries containing sample information
        """

        # Create empty set of samples
        samples_in_batch = []

        start = self._index_in_epoch
        # Shuffle only for the first epoch
        if self.epochs_completed == 0 and start == 0 and shuffle:
            self._shuffle_samples()

        # Go to the next epoch
        if start + batch_size >= self.num_samples:

            # Finished epoch
            self.epochs_completed += 1

            # Get the rest examples in this epoch
            rest_num_examples = self.num_samples - start

            # Append those samples to the current batch
            samples_in_batch.extend(
                self.get_sample_dict(np.arange(start, self.num_samples)))

            # Shuffle the data
            if shuffle:
                self._shuffle_samples()

            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch

            # Append the rest of the batch
            samples_in_batch.extend(self.get_sample_dict(np.arange(start, end)))

        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch

            # Append the samples in the range to the batch
            samples_in_batch.extend(self.get_sample_dict(np.arange(start, end)))

        return samples_in_batch
