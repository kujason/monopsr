import cv2

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from tf_ops.nn_distance import tf_nndistance
from tf_ops.approxmatch import tf_approxmatch

from monopsr.builders import loss_builder, net_builder
from monopsr.core import constants
from monopsr.core import img_preprocessor
from monopsr.core import orientation_encoder
from monopsr.core.models import model
from monopsr.core.models.monopsr import monopsr_output_builder
from monopsr.datasets.kitti import instance_utils


class MonoPSRModel(model.Model):

    def __init__(self, model_config, train_val_test, dataset):

        super(MonoPSRModel, self).__init__()

        self.model_config = model_config
        self.train_val_test = train_val_test
        self.dataset = dataset

        # TODO: parse model config
        self.image_input_shape = self.model_config.image_input_shape

        self.num_boxes = self.dataset.num_boxes
        self.num_alpha_bins = self.dataset.num_alpha_bins
        self.centroid_type = self.dataset.centroid_type
        self.depth_range = self.dataset.dataset_config.obj_filter_config.depth_range

        self.img_roi_size = model_config.img_roi_size
        self.map_roi_size = model_config.map_roi_size
        self.inst_crop_type = model_config.inst_crop_type
        self.resized_full_img_shape = model_config.resized_full_img_shape
        self.rotate_view = model_config.rotate_view
        self.use_pointnet_features = model_config.use_pointnet_features
        self.normalize_appended_ests = model_config.normalize_appended_ests

        self.net_type = model_config.net_type

        # Output config
        self.output_config = model_config.output_config
        self.output_types = monopsr_output_builder.MonoPSROutputBuilder.get_output_types_list(
            self.output_config)

        if not self.output_types:
            raise ValueError('output_types should not be empty')

        self.is_training = train_val_test == 'train'

        # Parse dataset config
        self.dataset_config = dataset.dataset_config

        # Set up preprocessor
        self.img_preprocessor = img_preprocessor.ImgPreprocessor()
        self.mean_sub_type = model_config.mean_sub_type

        # Set up input placeholders
        self._set_up_input_pls()

        # Post processing options
        self.post_process_cen_x = model_config.post_process_cen_x

    def _set_up_input_pls(self):
        """Sets up input placeholders
        """

        with tf.variable_scope('placeholders'):

            # Input
            self.pl_rgb_image = tf.placeholder(
                tf.float32, shape=[None, None, 3], name='rgb_image')
            self.pl_cam_p = tf.placeholder(
                tf.float32, shape=[3, 4], name='cam_p')
            self.pl_boxes_2d = tf.placeholder(
                tf.float32, shape=[self.num_boxes, 4], name='boxes_2d')
            self.pl_boxes_2d_norm = tf.placeholder(
                tf.float32, shape=[self.num_boxes, 4], name='boxes_2d_norm')
            self.pl_instance_masks = tf.placeholder(
                tf.float32, shape=[self.num_boxes, None, None], name='instance_masks')
            self.pl_class_strs = tf.placeholder(
                tf.string, shape=[self.num_boxes, 1], name='class_strs')
            self.pl_class_indices = tf.placeholder(
                tf.int32, shape=[self.num_boxes, 1], name='class_indices')
            self.pl_mean_lwh = tf.placeholder(
                tf.float32, shape=[self.num_boxes, 3], name='mean_lwh')

            # Initial estimates
            self.pl_prop_cen_z_offset = tf.placeholder(
                tf.float32, shape=[self.num_boxes], name='prop_cen_z_offset')

            self.pl_est_view_angs = tf.placeholder(
                tf.float32, shape=[self.num_boxes], name='est_view_angs')

            if self.train_val_test in ['train', 'val']:
                # Input
                self.pl_num_objs = tf.placeholder(
                    tf.int32, shape=[], name='num_objs')

                self.pl_depth_map = tf.placeholder(
                    tf.float32, shape=[None, None], name='depth_map')

                # Ground truth
                self.pl_gt_boxes_3d = tf.placeholder(
                    tf.float32, shape=[self.num_boxes, 7], name='boxes_3d')
                self.pl_gt_alphas = tf.placeholder(
                    tf.float32, shape=[self.num_boxes], name='alphas')
                self.pl_gt_alpha_bins = tf.placeholder(
                    tf.int32, shape=[self.num_boxes], name='alpha_bins')
                self.pl_gt_alpha_regs = tf.placeholder(
                    tf.float32, shape=[self.num_boxes, self.num_alpha_bins], name='alpha_regs')
                self.pl_gt_alpha_valid_bins = tf.placeholder(
                    tf.float32, shape=[self.num_boxes, self.num_alpha_bins], name='alpha_regs')
                self.pl_gt_view_angs = tf.placeholder(
                    tf.float32, shape=[self.num_boxes], name='view_angs')

            elif self.train_val_test in ['test']:
                pass
            else:
                raise ValueError('Invalid run mode', self.train_val_test)

        with tf.variable_scope('preprocess'):
            # Add batch dimension
            self.rgb_image_batched = tf.expand_dims(self.pl_rgb_image, axis=0)
            self.img_preprocessed = self.img_preprocessor.preprocess_input(
                self.rgb_image_batched, self.image_input_shape, mean_sub_type=self.mean_sub_type)

            if self.train_val_test in ['train', 'val']:
                self.tf_depth_map_batched = tf.expand_dims(self.pl_depth_map, axis=2)
                self.tf_depth_map_batched = tf.expand_dims(self.tf_depth_map_batched, axis=0)

    def build(self):
        is_training = self.train_val_test == 'train'
        is_train_or_val = self.train_val_test in ['train', 'val']

        tf_boxes_2d = self.pl_boxes_2d
        tf_instance_masks = self.pl_instance_masks

        tf_view_angs_2d = self.pl_est_view_angs
        cam_p = self.pl_cam_p

        if self.rotate_view:
            print('Rotate during view normalization')
        else:
            print('No rotation during view normalization')

        if is_train_or_val:

            tf_boxes_3d = self.pl_gt_boxes_3d
            tf_depth_map_batched = self.tf_depth_map_batched

            # Ground truth instance crops
            with tf.variable_scope('instance_crop'):
                if self.inst_crop_type == 'xyz_map':
                    raise NotImplementedError('No longer implemented')

                elif self.inst_crop_type == 'depth_map':
                    # TODO: These can be done together
                    # Get local instance xyz maps
                    gt_inst_xyz_maps_local_list = np.asarray([
                        instance_utils.tf_instance_xyz_crop_from_depth_map(
                            box_idx=box_idx,
                            tf_boxes_2d=tf_boxes_2d,
                            tf_boxes_3d=tf_boxes_3d,
                            tf_instance_masks=tf_instance_masks,
                            tf_depth_map_batched=tf_depth_map_batched,
                            roi_size=self.map_roi_size,
                            tf_viewing_angles=tf_view_angs_2d,
                            cam_p=cam_p,
                            view_norm=True,
                            centroid_type=self.centroid_type,
                            rotate_view=self.rotate_view)
                        for box_idx in np.arange(0, self.num_boxes, dtype=np.int32)])

                    # Get global instance xyz maps
                    gt_inst_xyz_maps_global_list = np.asarray([
                        instance_utils.tf_instance_xyz_crop_from_depth_map(
                            box_idx=box_idx,
                            tf_boxes_2d=tf_boxes_2d,
                            tf_boxes_3d=tf_boxes_3d,
                            tf_instance_masks=tf_instance_masks,
                            tf_depth_map_batched=tf_depth_map_batched,
                            roi_size=self.map_roi_size,
                            tf_viewing_angles=tf_view_angs_2d,
                            cam_p=cam_p,
                            view_norm=False,
                            centroid_type=self.centroid_type,
                            rotate_view=self.rotate_view)
                        for box_idx in np.arange(0, self.num_boxes, dtype=np.int32)])

                else:
                    raise ValueError('Invalid inst_crop_type')

                gt_inst_xyz_maps_local = tf.concat(
                    list(gt_inst_xyz_maps_local_list[:, 0]), axis=0)
                gt_inst_xyz_maps_global = tf.concat(
                    list(gt_inst_xyz_maps_global_list[:, 0]), axis=0)

                gt_valid_mask_maps = tf.concat(list(gt_inst_xyz_maps_local_list[:, 1]), axis=0)
        else:
            gt_inst_xyz_maps_local = None
            gt_inst_xyz_maps_global = None

            tf_boxes_3d = None

            # Crop and resize instance masks
            # valid_mask_maps_list = np.asarray([instance_utils.tf_crop_and_resize_instance_masks(
            #     self.pl_instance_masks, tf_boxes_2d, self.map_roi_size, box_idx) for
            #     box_idx in np.arange(0, self.num_boxes, dtype=np.int32)])
            # valid_mask_maps = tf.concat(list(valid_mask_maps_list), axis=0)

            gt_valid_mask_maps = tf.ones([1, *self.map_roi_size, 1])

        # Inputs
        input_dict = {}
        with tf.variable_scope('image_crop'):
            rgb_image_crops = tf.image.crop_and_resize(
                self.img_preprocessed, self.pl_boxes_2d_norm,
                tf.zeros(self.num_boxes, dtype=tf.int32), self.img_roi_size)
            input_dict[constants.NET_IN_RGB_CROP] = rgb_image_crops

        with tf.variable_scope('full_img'):
            # Resize image used for full scene features to reduce memory in net
            full_img_resized = tf.image.resize_bilinear(self.img_preprocessed,
                                                        self.resized_full_img_shape,
                                                        align_corners=True)
            input_dict[constants.NET_IN_FULL_IMG] = full_img_resized

        # Net builder
        features_dict = net_builder.extract_features(
            self, self.net_type, self.model_config, input_dict, is_training)

        return self.build_outputs(
            features_dict=features_dict,
            gt_valid_mask_maps=gt_valid_mask_maps,
            gt_inst_xyz_maps_local=gt_inst_xyz_maps_local,
            gt_inst_xyz_maps_global=gt_inst_xyz_maps_global,
            tf_boxes_2d=tf_boxes_2d,
            tf_boxes_3d=tf_boxes_3d,
        )

    def build_outputs(self, features_dict,
                      gt_valid_mask_maps,
                      gt_inst_xyz_maps_local,
                      gt_inst_xyz_maps_global,
                      tf_boxes_2d, tf_boxes_3d):

        # DEBUG
        output_debug_dict = {}

        # Output
        with tf.variable_scope('output'):
            print('Building outputs:')

            # Setup ground truth
            with tf.variable_scope('box_gt'):
                if self.train_val_test in ['train', 'val']:
                    gt_cen_x = tf_boxes_3d[:, 0:1]

                    if self.centroid_type == 'middle':
                        # Move y centroid to middle of 3D box
                        gt_h = tf_boxes_3d[:, 5:6]
                        gt_bottom_cen_y = tf_boxes_3d[:, 1:2]
                        gt_cen_y = gt_bottom_cen_y - gt_h / 2
                    elif self.centroid_type == 'bottom':
                        gt_cen_y = tf_boxes_3d[:, 1:2]
                    else:
                        raise ValueError('Invalid centroid type', self.centroid_type)

                    gt_cen_z = tf_boxes_3d[:, 2:3]
                    gt_centroids = tf.concat([gt_cen_x, gt_cen_y, gt_cen_z], axis=1)

                    gt_lwh = tf_boxes_3d[:, 3:6]
                    gt_alpha = tf.expand_dims(self.pl_gt_alphas, axis=1)
                    gt_alpha_dc = [tf.expand_dims(self.pl_gt_alpha_bins, axis=1),
                                   self.pl_gt_alpha_regs]
                    gt_view_angs = tf.expand_dims(self.pl_gt_view_angs, axis=1)
                else:
                    # Test mode
                    gt_lwh = None
                    gt_alpha = None
                    gt_alpha_dc = None
                    gt_cen_z = None
                    gt_view_angs = None
                    gt_cen_y = None
                    gt_centroids = None

            # Create output builder
            model_output_builder = monopsr_output_builder.MonoPSROutputBuilder(
                self.output_config, self.model_config, self.dataset_config,
                features_dict, self.num_boxes, self.map_roi_size, self.pl_cam_p,
                train_val_test=self.train_val_test)

            output_dict = model_output_builder.get_output_dict()
            gt_dict = model_output_builder.get_gt_dict()

            # Local instance maps
            if constants.KEY_INST_XYZ_MAP_LOCAL in self.output_types:
                model_output_builder.add_inst_xyz_maps_local(
                    gt_inst_xyz_maps_local=gt_inst_xyz_maps_local)

            # Valid mask maps
            model_output_builder._gt_dict.add_unique_to_dict({
                constants.KEY_VALID_MASK_MAPS: gt_valid_mask_maps,
            })
            if constants.KEY_VALID_MASK_MAPS in self.output_types:
                model_output_builder.add_valid_mask_maps_output()
            else:
                # Use ground truth masks
                output_dict.add_unique_to_dict({
                    constants.KEY_VALID_MASK_MAPS: gt_valid_mask_maps,
                })

            with tf.variable_scope('expand_dims'):
                est_view_angs = tf.expand_dims(self.pl_est_view_angs, axis=1)

            # Fully connected layers for box regression
            model_output_builder.add_proposal_fc_features(
                boxes_2d=tf_boxes_2d,
                view_angs=est_view_angs,
                class_indices=self.pl_class_indices,
                image_shape=self.image_input_shape)
            init_est_fc_features = model_output_builder.get_proposal_fc_features()

            # Add lwh outputs
            if constants.KEY_LWH in self.output_types:
                pred_lwh = model_output_builder.add_lwh_output(
                    features_to_use=init_est_fc_features,
                    est_lwh=self.pl_mean_lwh,
                    gt_lwh=gt_lwh)

            # Add alpha outputs
            if constants.KEY_ALPHA in self.output_types:
                model_output_builder.add_alpha_output(
                    features_to_use=init_est_fc_features,
                    gt_alpha=gt_alpha,
                    gt_alpha_dc=gt_alpha_dc)

            # Add viewing angle outputs
            if constants.KEY_VIEW_ANG in self.output_types:
                model_output_builder.add_view_ang_output(
                    output_key=constants.KEY_VIEW_ANG,
                    features_in=init_est_fc_features,
                    est_view_angs=est_view_angs,
                    gt_view_angs=gt_view_angs)

            # Get centroid proposal
            prop_cen_z = model_output_builder.get_prop_cen_z(
                tf_boxes_2d, self.pl_prop_cen_z_offset)
            prop_cen_y = model_output_builder.get_prop_cen_y(
                tf_boxes_2d, prop_cen_z, self.dataset.classes_name)

            # Setup regression layers
            est_lwh_offs = output_dict[constants.KEY_LWH + '_offs']
            est_alpha_bins = output_dict[constants.KEY_ALPHA_BINS]
            est_alpha_regs = output_dict[constants.KEY_ALPHA_REGS]
            max_depth = self.depth_range[1]

            model_output_builder.add_regression_fc_features(
                boxes_2d=tf_boxes_2d,
                view_angs=est_view_angs,
                class_indices=self.pl_class_indices,
                image_shape=self.image_input_shape,

                est_lwh_off=est_lwh_offs,
                est_alpha_bins=est_alpha_bins,
                est_alpha_regs=est_alpha_regs,
                prop_cen_y=prop_cen_y,
                prop_cen_z=prop_cen_z,
                max_depth=max_depth)
            regression_fc_features = model_output_builder.get_regression_fc_features()

            # Add regression from proposed position
            model_output_builder.add_cen_y_output(
                output_key=constants.KEY_CEN_Y,
                features_in=regression_fc_features,
                prop_cen_y=prop_cen_y,
                gt_cen_y=gt_cen_y)

            model_output_builder.add_cen_z_output(
                output_key=constants.KEY_CEN_Z,
                features_in=regression_fc_features,
                prop_cen_z=prop_cen_z,
                gt_cen_z=gt_cen_z)

            # Add centroid x output
            if constants.KEY_CEN_X in self.output_types:
                pred_cen_z = output_dict[constants.KEY_CEN_Z]
                pred_view_angs = output_dict[constants.KEY_VIEW_ANG]

                model_output_builder.add_cen_x_output(
                    output_key=constants.KEY_CEN_X,
                    pred_cen_z=pred_cen_z,
                    pred_view_angs=pred_view_angs)

            # Add centroid output
            if constants.KEY_CENTROIDS in self.output_types:
                pred_cen_x = output_dict[constants.KEY_CEN_X]
                pred_cen_y = output_dict[constants.KEY_CEN_Y]
                pred_cen_z = output_dict[constants.KEY_CEN_Z]

                model_output_builder.add_centroids_output(
                    output_key=constants.KEY_CENTROIDS,
                    pred_cen_x=pred_cen_x,
                    pred_cen_y=pred_cen_y,
                    pred_cen_z=pred_cen_z,
                    gt_centroids=gt_centroids)

            # # # Global Maps # # #
            if self.train_val_test in ['train', 'val']:

                if constants.KEY_INST_XYZ_MAP_GLOBAL in self.output_types:
                    with tf.variable_scope('xyz_projection'):
                        pred_inst_xyz_maps_local = output_dict[constants.KEY_INST_XYZ_MAP_LOCAL]

                        # Calculate x from gt viewing angle
                        pred_cen_y = output_dict[constants.KEY_CEN_Y]
                        pred_cen_z = output_dict[constants.KEY_CEN_Z]
                        proj_cam2_pred_cen_x = pred_cen_z * tf.tan(gt_view_angs)

                        # Adjust for x_offset
                        cam_p = self.pl_cam_p
                        x_offset = -cam_p[0, 3] / cam_p[0, 0]
                        proj_gt_cen_x = proj_cam2_pred_cen_x + x_offset

                        proj_pred_cen = tf.concat(
                            [proj_gt_cen_x, pred_cen_y, pred_cen_z], axis=1)

                        # Place at gt centroid position?
                        pred_inst_xyz_maps_global = model_output_builder.get_inst_xyz_map_global(
                            pred_inst_xyz_maps_local=pred_inst_xyz_maps_local,
                            pred_view_angs=gt_view_angs,
                            pred_centroids=proj_pred_cen)

                        proj_err_norm, proj_err_debug_dict = \
                            model_output_builder.get_proj_err_maps_norm(
                                pred_inst_xyz_map_global=pred_inst_xyz_maps_global,
                                pred_boxes_2d=tf_boxes_2d,
                                valid_mask_maps=gt_valid_mask_maps)

                        output_dict.add_unique_to_dict({'proj_err_norm': proj_err_norm})

                if constants.KEY_INST_DEPTH_MAP_GLOBAL in self.output_types:

                    if constants.KEY_INST_XYZ_MAP_LOCAL in self.output_types:
                        # Use local xyz map
                        with tf.variable_scope('depth_maps_global'):
                            pred_inst_xyz_map_local = output_dict[constants.KEY_INST_XYZ_MAP_LOCAL]
                            pred_inst_depth_map_local = pred_inst_xyz_map_local[:, :, :, 2:3]
                            gt_inst_depth_maps_global = gt_inst_xyz_maps_global[:, :, :, 2:3]

                        model_output_builder.add_inst_depth_maps_global(
                            pred_inst_depth_maps_local=pred_inst_depth_map_local,
                            gt_inst_depth_maps_global=gt_inst_depth_maps_global,
                            rotate_view=self.rotate_view, box_2d=tf_boxes_2d)

        # Add class indices to output_dict
        output_dict = model_output_builder.get_output()
        output_dict.update({
            constants.SAMPLE_LABEL_CLASS_INDICES: self.pl_class_indices
        })

        print('Done building outputs')

        # # DEBUG
        # output_debug_dict.update({
        #     # 'roi_crops': roi_crops,
        #     # 'rgb_image_crops': rgb_image_crops,
        #     'valid_mask_maps': valid_mask_maps,
        #     'gt_inst_xyz_maps_local': gt_inst_xyz_maps_local,
        #     'gt_inst_xyz_maps_global': gt_inst_xyz_maps_global,
        #     'pred_inst_xyz_maps_local': pred_inst_xyz_maps_local,
        #     # 'pred_inst_xyz_maps_global': pred_inst_xyz_maps_global,
        #     'pred_inst_xyz_maps_global': pred_inst_xyz_maps_global,
        #
        #     'pred_view_angs': pred_view_angs,
        #     'gt_view_angs': gt_view_angs,
        #
        #     'pred_cen_x': pred_cen_x,
        #     'pred_cen_z': pred_cen_z,
        #     'gt_pred_cen_x': pred_cen_z * tf.tan(gt_view_angs),
        #     'proj_pred_cen': proj_pred_cen,
        # })
        # output_debug_dict.update(proj_err_debug_dict)

        return output_dict, gt_dict, output_debug_dict

    def create_feed_dict(self):

        # TODO: don't skip empty samples during evaluation/testing
        sample_dict = None
        while sample_dict is None:

            if self.train_val_test == 'train':
                sample_dict = self.dataset.next_batch(batch_size=1, shuffle=True)[0]
            else:
                sample_dict = self.dataset.next_batch(batch_size=1, shuffle=False)[0]

        # Create feed dict
        feed_dict = {}

        feed_dict.update({
            self.pl_rgb_image: sample_dict[constants.SAMPLE_IMAGE_INPUT],

            self.pl_boxes_2d: sample_dict[constants.SAMPLE_LABEL_BOXES_2D],
            self.pl_boxes_2d_norm: sample_dict[constants.SAMPLE_LABEL_BOXES_2D_NORM],

            self.pl_class_strs: sample_dict[constants.SAMPLE_LABEL_CLASS_STRS],
            self.pl_class_indices: sample_dict[constants.SAMPLE_LABEL_CLASS_INDICES],
            self.pl_mean_lwh: sample_dict[constants.SAMPLE_MEAN_LWH],

            self.pl_cam_p: sample_dict[constants.SAMPLE_CAM_P],

            self.pl_prop_cen_z_offset: sample_dict[constants.SAMPLE_PROP_CEN_Z_OFFSET],

            self.pl_est_view_angs: sample_dict[constants.SAMPLE_VIEWING_ANGLES_2D],

        })

        if self.train_val_test in ['train', 'val']:
            feed_dict.update({
                self.pl_num_objs: sample_dict[constants.SAMPLE_NUM_OBJS],

                self.pl_gt_boxes_3d: sample_dict[constants.SAMPLE_LABEL_BOXES_3D],

                self.pl_instance_masks: sample_dict[constants.SAMPLE_INSTANCE_MASKS],
                self.pl_gt_alphas: sample_dict[constants.SAMPLE_ALPHAS],

                self.pl_gt_alpha_bins: sample_dict[constants.SAMPLE_ALPHA_BINS],
                self.pl_gt_alpha_regs: sample_dict[constants.SAMPLE_ALPHA_REGS],
                self.pl_gt_alpha_valid_bins: sample_dict[constants.SAMPLE_ALPHA_VALID_BINS],

                self.pl_gt_view_angs: sample_dict[constants.SAMPLE_VIEWING_ANGLES_3D],

                self.pl_depth_map: sample_dict[constants.SAMPLE_DEPTH_MAP],

            })

        elif self.train_val_test in ['test']:
            # No additional inputs for test mode
            pass

        else:
            raise ValueError('Invalid run mode', self.train_val_test)

        return feed_dict, sample_dict

    def loss(self, output_dict, gt_dict):
        print('Building losses: ')

        with tf.variable_scope('losses'):

            with tf.variable_scope('total_loss'):
                total_loss = 0

            # Get loss config
            loss_config = self.model_config.loss_config

            # Shared loss weighting
            loss_mask_ones = tf.ones([1, self.num_boxes, 1], tf.float32)

            # TODO: Move into loss builder
            # Setup losses dict
            losses_dict = {}
            if constants.KEY_INST_XYZ_MAP_LOCAL in self.output_types:

                with tf.variable_scope('xyz_map'):
                    # Parse prediction dict
                    pred_inst_xyz_map_local = output_dict[constants.KEY_INST_XYZ_MAP_LOCAL]

                    gt_inst_xyz_map_local = gt_dict[constants.KEY_INST_XYZ_MAP_LOCAL]
                    gt_valid_mask_maps = gt_dict[constants.KEY_VALID_MASK_MAPS]

                    inst_xyz_map_local_loss = loss_builder.add_loss_tensor(
                        loss_config, constants.KEY_INST_XYZ_MAP_LOCAL,
                        pred_inst_xyz_map_local, gt_inst_xyz_map_local,
                        mask=gt_valid_mask_maps)

                    inst_xyz_map_local_loss_norm = inst_xyz_map_local_loss / self.num_boxes

                    total_loss += inst_xyz_map_local_loss_norm

                losses_dict.update({
                    constants.KEY_INST_XYZ_MAP_LOCAL: inst_xyz_map_local_loss_norm,
                })

                # Add summary with scope 'losses/<name>'
                tf.summary.scalar(
                    constants.KEY_INST_XYZ_MAP_LOCAL, inst_xyz_map_local_loss_norm)

            if constants.KEY_VALID_MASK_MAPS in self.output_types:
                with tf.variable_scope('valid_mask_maps'):
                    pred_valid_mask_maps = output_dict[constants.KEY_VALID_MASK_MAPS]
                    gt_valid_mask_maps = gt_dict[constants.KEY_VALID_MASK_MAPS]

                    gt_valid_mask_maps_smoothed = gt_valid_mask_maps * 0.998 + 0.001

                    mask_map_loss = loss_builder.add_loss_tensor(
                        loss_config, constants.KEY_VALID_MASK_MAPS,
                        pred_valid_mask_maps, gt_valid_mask_maps_smoothed,
                        tf.ones_like(gt_valid_mask_maps))

                    # Normalize by number of pixels
                    num_pixels = self.map_roi_size[0] * self.map_roi_size[1]
                    num_pixels = tf.to_float(tf.expand_dims([num_pixels], axis=1))
                    mask_loss_sum = tf.reduce_sum(mask_map_loss, axis=[1, 2])
                    mask_loss_norm = tf.reduce_sum(mask_loss_sum / num_pixels)

                    # Add to total loss
                    total_loss += mask_loss_norm

                # Update loss dict and add to summaries with scope 'losses/<name>'
                losses_dict.update({
                    constants.KEY_VALID_MASK_MAPS: mask_loss_norm})
                tf.summary.scalar(
                    constants.KEY_VALID_MASK_MAPS, mask_loss_norm)

            if constants.KEY_LWH in self.output_types:
                loss_key_offs = constants.KEY_LWH + '_offs'

                with tf.variable_scope('dimensions'):
                    # Parse prediction dict
                    pred_dim_offsets = tf.expand_dims(
                        output_dict[loss_key_offs], axis=0)

                    gt_dim_offsets = tf.expand_dims(
                        gt_dict[loss_key_offs], axis=0)

                    # Dim loss calculation
                    dim_reg_loss = loss_builder.add_loss_tensor(
                        loss_config, constants.KEY_LWH,
                        pred_dim_offsets, gt_dim_offsets, mask=loss_mask_ones)
                    dim_reg_loss = tf.reduce_sum(dim_reg_loss)

                    # Normalize
                    dim_reg_loss_norm = dim_reg_loss / self.num_boxes

                    # Add to total loss
                    total_loss += dim_reg_loss_norm

                # Update loss dict and add to summaries
                losses_dict.update({
                    loss_key_offs: dim_reg_loss_norm,
                })
                tf.summary.scalar(loss_key_offs, dim_reg_loss_norm)

            if constants.KEY_ALPHA in self.output_types:

                output_type = self.output_config.alpha

                if output_type in ['dc', 'dc_rotation']:

                    with tf.variable_scope('alpha'):

                        # Parse prediction dict
                        pred_alpha_bins = tf.expand_dims(
                            output_dict[constants.KEY_ALPHA_BINS], axis=0)
                        pred_alpha_regs = tf.expand_dims(
                            output_dict[constants.KEY_ALPHA_REGS], axis=0)

                        # Get label smoothing epsilon
                        alpha_config = getattr(loss_config, constants.KEY_ALPHA + '_cls')
                        label_smoothing_eps = alpha_config[2]

                        # Get gt bins and regressions
                        gt_alpha_bins = tf.expand_dims(
                            gt_dict[constants.KEY_ALPHA_BINS], axis=0)
                        gt_alpha_bins_one_hot = tf.squeeze(
                            tf.one_hot(gt_alpha_bins, self.num_alpha_bins,
                                       on_value=1.0 - label_smoothing_eps,
                                       off_value=label_smoothing_eps / self.dataset.num_alpha_bins),
                            axis=2)
                        gt_alpha_regs = tf.expand_dims(
                            gt_dict[constants.KEY_ALPHA_REGS], axis=0)

                        # Alpha bins loss
                        alpha_bins_loss = loss_builder.add_loss_tensor(
                            loss_config, constants.KEY_ALPHA + '_cls',
                            pred_alpha_bins, gt_alpha_bins_one_hot, mask=loss_mask_ones)
                        alpha_bins_loss = tf.reduce_sum(alpha_bins_loss)

                        # Alpha regression loss
                        alpha_bin_mask = tf.expand_dims(self.pl_gt_alpha_valid_bins, 0)
                        alpha_reg_loss = loss_builder.add_loss_tensor(
                            loss_config, constants.KEY_ALPHA + '_reg',
                            pred_alpha_regs, gt_alpha_regs, mask=alpha_bin_mask)
                        alpha_reg_loss = tf.reduce_sum(alpha_reg_loss)

                        # Normalize by number of boxes
                        alpha_bins_loss_norm = alpha_bins_loss / self.num_boxes
                        alpha_reg_loss_norm = alpha_reg_loss / self.num_boxes

                        # Add to total loss
                        alpha_loss = alpha_bins_loss_norm + alpha_reg_loss_norm
                        total_loss += alpha_loss

                    # Update loss dict and add to summaries
                    losses_dict.update({
                        constants.KEY_ALPHA_BINS: alpha_bins_loss_norm,
                        constants.KEY_ALPHA_REGS: alpha_reg_loss_norm,
                    })

                    tf.summary.scalar(constants.KEY_ALPHA_BINS, alpha_bins_loss_norm)
                    tf.summary.scalar(constants.KEY_ALPHA_REGS, alpha_reg_loss_norm)

                elif output_type == 'prob':

                    with tf.variable_scope('alpha'):
                        # Parse prediction dict
                        pred_alpha_bins = tf.expand_dims(
                            output_dict[constants.KEY_ALPHA_BINS], axis=0)
                        pred_alphas = tf.expand_dims(
                            output_dict[constants.KEY_ALPHA], axis=0)

                        # Get gt bins and alphas
                        gt_alpha_bins = tf.expand_dims(
                            gt_dict[constants.KEY_ALPHA_BINS], axis=0)
                        gt_alpha_bins_one_hot = tf.squeeze(
                            tf.one_hot(gt_alpha_bins, self.num_alpha_bins), axis=2)
                        gt_alphas = tf.expand_dims(
                            gt_dict[constants.KEY_ALPHA], axis=0)

                        # Alpha bins loss
                        alpha_bins_loss = loss_builder.add_loss_tensor(
                            loss_config, constants.KEY_ALPHA + '_cls_temp',
                            pred_alpha_bins, gt_alpha_bins_one_hot, mask=loss_mask_ones)
                        alpha_bins_loss = tf.reduce_sum(alpha_bins_loss)

                        # Alpha reg loss
                        alpha_reg_loss = loss_builder.add_loss_tensor(
                            loss_config, constants.KEY_ALPHA + '_reg',
                            pred_alphas, gt_alphas, mask=loss_mask_ones)
                        alpha_reg_loss = tf.reduce_sum(alpha_reg_loss)

                        # Normalize by number of boxes
                        alpha_bins_loss_norm = alpha_bins_loss / self.num_boxes
                        alpha_reg_loss_norm = alpha_reg_loss / self.num_boxes

                        # Add to total loss
                        alpha_loss = alpha_bins_loss_norm + alpha_reg_loss_norm
                        total_loss += alpha_loss

                    # Update loss dict and add to summaries
                    losses_dict.update({
                        constants.KEY_ALPHA_BINS: alpha_bins_loss_norm,
                        constants.KEY_ALPHA: alpha_reg_loss_norm,
                    })

                    tf.summary.scalar(constants.KEY_ALPHA_BINS, alpha_bins_loss_norm)
                    tf.summary.scalar(constants.KEY_ALPHA, alpha_reg_loss_norm)

                elif output_type == 'gt':
                    # No losses to compute
                    pass

            if constants.KEY_CEN_Z in self.output_types:

                output_type = self.output_config.cen_z

                if output_type == 'offset':
                    loss_key_offs = constants.KEY_CEN_Z + '_offs'

                    with tf.variable_scope('cen_z_off'):
                        # Parse prediction dict
                        pred_cen_z_offs = tf.expand_dims(
                            output_dict[loss_key_offs], axis=0)

                        gt_cen_z_offs = tf.expand_dims(
                            gt_dict[loss_key_offs], axis=0)

                        # cen z reg loss calculation
                        cen_z_reg_loss = loss_builder.add_loss_tensor(
                            loss_config, constants.KEY_CEN_Z,
                            pred_cen_z_offs, gt_cen_z_offs, mask=loss_mask_ones)

                        cen_z_reg_loss = tf.reduce_sum(cen_z_reg_loss)
                        cen_z_reg_loss_norm = cen_z_reg_loss / self.num_boxes

                        total_loss += cen_z_reg_loss_norm

                    # Update loss dict and add to summaries
                    losses_dict.update({
                        loss_key_offs: cen_z_reg_loss_norm,
                    })
                    tf.summary.scalar(loss_key_offs, cen_z_reg_loss_norm)

                elif output_type in ['gt', 'est']:
                    # No losses to compute
                    pass

                else:
                    raise ValueError('Invalid output type:', output_type)

            if constants.KEY_VIEW_ANG in self.output_types:

                output_type = self.output_config.view_ang

                if output_type == 'offset':

                    loss_key = constants.KEY_VIEW_ANG
                    loss_key_offs = loss_key + '_offs'

                    with tf.variable_scope('viewing_angle'):
                        # Parse prediction dict
                        pred_view_ang_offs = tf.expand_dims(
                            output_dict[loss_key_offs], axis=0)

                        gt_view_ang_offs = tf.expand_dims(
                            gt_dict[loss_key_offs], axis=0)

                        # Viewing angle loss calculation
                        view_ang_off_loss = loss_builder.add_loss_tensor(
                            loss_config, loss_key,
                            pred_view_ang_offs, gt_view_ang_offs, mask=loss_mask_ones)
                        view_ang_off_loss = tf.reduce_sum(view_ang_off_loss)

                        # Normalize
                        view_ang_off_loss_norm = view_ang_off_loss / self.num_boxes

                        # Add to total loss
                        total_loss += view_ang_off_loss_norm

                    # Update loss dict and add to summaries
                    losses_dict.update({loss_key_offs: view_ang_off_loss_norm})
                    tf.summary.scalar(loss_key_offs, view_ang_off_loss_norm)

                elif output_type in ['est', 'gt']:
                    # No losses to compute
                    pass

            if constants.KEY_CEN_Y in self.output_types:

                output_type = self.output_config.cen_y

                if output_type == 'offset':

                    loss_key = constants.KEY_CEN_Y
                    loss_key_offs = loss_key + '_offs'

                    with tf.variable_scope('centroid_y'):
                        # Parse prediction dict
                        pred_cen_y_offs = tf.expand_dims(output_dict[loss_key_offs], axis=0)

                        gt_cen_y_offs = tf.expand_dims(gt_dict[loss_key_offs], axis=0)

                        # cen y reg loss calculation
                        cen_y_reg_loss = loss_builder.add_loss_tensor(
                            loss_config, loss_key,
                            pred_cen_y_offs, gt_cen_y_offs, mask=loss_mask_ones)
                        cen_y_reg_loss = tf.reduce_sum(cen_y_reg_loss)

                        # Normalize
                        cen_y_reg_loss_norm = cen_y_reg_loss / self.num_boxes

                        # Add to total loss
                        total_loss += cen_y_reg_loss_norm

                    # Update loss dict and add to summaries
                    losses_dict.update({
                        loss_key_offs: cen_y_reg_loss_norm,
                    })
                    tf.summary.scalar(loss_key_offs, cen_y_reg_loss_norm)

                elif output_type in ['gt', 'est']:
                    pass

                else:
                    raise ValueError('Invalid output_type:', output_type)

            if constants.KEY_INST_XYZ_MAP_GLOBAL in self.output_types:

                with tf.variable_scope('xyz_proj_err'):
                    # Projection error
                    proj_err_norm = tf.reshape(output_dict['proj_err_norm'], (1, -1, 1))
                    gt_proj_err = tf.zeros_like(proj_err_norm)

                    # Normalize
                    # gt_valid_mask_maps = gt_dict[constants.KEY_VALID_MASK_MAPS]
                    proj_err_loss = loss_builder.add_loss_tensor(
                        loss_config, constants.KEY_INST_XYZ_MAP_GLOBAL,
                        proj_err_norm, gt_proj_err, mask=loss_mask_ones)
                    # proj_err_norm, gt_proj_err, mask=gt_valid_mask_maps)

                    # Add to total loss
                    total_loss += proj_err_loss

                # Update loss dict and add to summaries
                losses_dict.update(
                    {'proj_err': proj_err_loss})
                tf.summary.scalar(
                    'proj_err', proj_err_loss)

            if constants.KEY_INST_DEPTH_MAP_GLOBAL in self.output_types:

                with tf.variable_scope('global_depth_map'):
                    # Parse prediction dict
                    pred_inst_depth_map_global = output_dict[constants.KEY_INST_DEPTH_MAP_GLOBAL]

                    gt_inst_depth_map_global = gt_dict[constants.KEY_INST_DEPTH_MAP_GLOBAL]
                    gt_valid_mask_maps = gt_dict[constants.KEY_VALID_MASK_MAPS]

                    # Calculate loss
                    inst_depth_map_global_loss = loss_builder.add_loss_tensor(
                        loss_config, constants.KEY_INST_DEPTH_MAP_GLOBAL,
                        pred_inst_depth_map_global,
                        gt_inst_depth_map_global,
                        mask=gt_valid_mask_maps)

                    # Normalize
                    inst_depth_map_global_loss_norm = inst_depth_map_global_loss / self.num_boxes

                    # Add to total loss
                    total_loss += inst_depth_map_global_loss_norm

                # Update loss dict and add to summaries
                losses_dict.update(
                    {constants.KEY_INST_DEPTH_MAP_GLOBAL: inst_depth_map_global_loss_norm})
                tf.summary.scalar(
                    constants.KEY_INST_DEPTH_MAP_GLOBAL, inst_depth_map_global_loss_norm)

            if constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH in self.output_types:

                with tf.variable_scope('global_xyz_map'):
                    # Parse prediction dict
                    pred_inst_global_xyz_map = \
                        output_dict[constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH]

                    gt_inst_xyz_map_global = gt_dict[constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH]
                    gt_valid_mask_maps = gt_dict[constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH]

                    # Calculate loss
                    inst_xyz_map_global_from_depth_loss = loss_builder.add_loss_tensor(
                        loss_config, constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH,
                        pred_inst_global_xyz_map, gt_inst_xyz_map_global,
                        mask=gt_valid_mask_maps)

                    # Normalize
                    inst_xyz_map_global_from_depth_loss_norm = \
                        inst_xyz_map_global_from_depth_loss / self.num_boxes

                    # Add to total loss
                    total_loss += inst_xyz_map_global_from_depth_loss_norm

                # Update loss dict and add to summaries
                losses_dict.update({
                    constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH:
                        inst_xyz_map_global_from_depth_loss_norm
                })
                tf.summary.scalar(constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH,
                                  inst_xyz_map_global_from_depth_loss_norm)

        return losses_dict, total_loss

    def format_predictions(self, output_types, output_dict, sample_dict):
        """Format predictions for saving and evaluating"""

        sample_name = sample_dict[constants.SAMPLE_NAME]
        img = sample_dict[constants.SAMPLE_IMAGE_INPUT]
        num_objs = sample_dict[constants.SAMPLE_NUM_OBJS]
        cam_p = sample_dict[constants.SAMPLE_CAM_P]
        all_scores = sample_dict[constants.SAMPLE_LABEL_SCORES]
        valid_scores = np.expand_dims(all_scores[0:num_objs], 1)
        valid_mask_maps = output_dict[constants.KEY_VALID_MASK_MAPS][0:num_objs]

        pred_dict = {}

        # Convert to float mask (outputs are logits trained with sigmoid loss)
        valid_mask_maps = (valid_mask_maps > 0.0).astype(np.float32)
        pred_dict[constants.KEY_VALID_MASK_MAPS] = valid_mask_maps

        if constants.KEY_INST_XYZ_MAP_LOCAL in output_types:
            pred_inst_xyz_maps_local = output_dict[constants.KEY_INST_XYZ_MAP_LOCAL][0:num_objs]
            pred_inst_xyz_maps_local_masked = pred_inst_xyz_maps_local * valid_mask_maps
            pred_dict[constants.KEY_INST_XYZ_MAP_LOCAL] = pred_inst_xyz_maps_local_masked

        if constants.KEY_CENTROIDS in output_types:

            # Get detected 2D boxes
            new_boxes_2d = np.copy(sample_dict[constants.SAMPLE_LABEL_BOXES_2D])

            if self.train_val_test in ['train', 'val']:
                # Get ground truth box_3d prediction and replace values if predicted
                new_boxes_3d = np.copy(sample_dict[constants.SAMPLE_LABEL_BOXES_3D])
            elif self.train_val_test == 'test':
                new_boxes_3d = np.zeros([self.num_boxes, 7], dtype=np.float32)
            else:
                raise ValueError('Invalid run mode', self.train_val_test)

            if constants.KEY_LWH in self.output_types:
                # Get dimension offset predictions and convert to full dimensions
                pred_lwh = output_dict[constants.KEY_LWH]

                # Add predictions to box 3d prediction
                new_boxes_3d[:, 3:6] = pred_lwh

            # Get sample viewing angle
            if constants.KEY_VIEW_ANG in self.output_types:
                sample_viewing_angles = output_dict[constants.KEY_VIEW_ANG]
            else:
                sample_viewing_angles = sample_dict[constants.SAMPLE_VIEWING_ANGLES_3D]

            if constants.KEY_ALPHA in self.output_types:

                alpha_type = getattr(self.output_config, constants.KEY_ALPHA)

                if alpha_type in ['dc', 'dc_rotation', 'gt']:

                    # Get orientation predictions
                    pred_alpha_bins = output_dict[constants.KEY_ALPHA_BINS]
                    pred_alpha_regs = output_dict[constants.KEY_ALPHA_REGS]
                    best_pred_alpha_bins = np.argmax(pred_alpha_bins, axis=1)
                    best_pred_alpha_regs = [pred_alpha_regs[i, ang_bin]
                                            for i, ang_bin in enumerate(best_pred_alpha_bins)]

                    pred_alphas = [orientation_encoder.np_angle_bin_to_orientation(
                        ang_bin, ang_reg, self.num_alpha_bins)
                        for (ang_bin, ang_reg) in zip(best_pred_alpha_bins, best_pred_alpha_regs)]

                elif alpha_type == 'prob':
                    pred_alphas = np.squeeze(output_dict[constants.KEY_ALPHA])

                else:
                    raise ValueError('Invalid alpha_type', alpha_type)

                # Convert alpha angles to global rotation
                pred_rys = pred_alphas + np.squeeze(sample_viewing_angles)

                # Add predictions to box 3d prediction
                new_boxes_3d[:, 6] = pred_rys

            else:
                pred_alphas = new_boxes_3d[:, 6] - np.squeeze(sample_viewing_angles)

            pred_centroids = output_dict[constants.KEY_CENTROIDS]

            if self.centroid_type == 'middle':
                pred_half_height = new_boxes_3d[:, 5:6] / 2
                pred_centroids[:, 1:2] = pred_centroids[:, 1:2] + pred_half_height
            new_boxes_3d[:, 0:3] = pred_centroids

            if self.post_process_cen_x:
                new_cen_x = np.asarray([instance_utils.postprocess_cen_x(box_2d, box_3d, cam_p)
                                        for box_2d, box_3d, in zip(new_boxes_2d, new_boxes_3d)])
                new_boxes_3d[:, 0] = np.squeeze(new_cen_x)

            # Mask predictions to number of boxes
            valid_boxes_3d = new_boxes_3d[0:num_objs]
            valid_boxes_2d = new_boxes_2d[0:num_objs]

            # Get score
            new_valid_scores = monopsr_output_builder.score_boxes(
                self.dataset, sample_name, img.shape, valid_boxes_2d, valid_boxes_3d, valid_scores)

            # Get class from detection. Subtract 1 for proper indexing with dataset.classes
            classes = output_dict[constants.SAMPLE_LABEL_CLASS_INDICES][0:num_objs] - 1
            detections_box_3d = np.hstack([valid_boxes_3d, new_valid_scores, classes])
            pred_dict[constants.KEY_BOX_3D] = detections_box_3d

            # Save box 2d parameters
            valid_pred_alphas = np.expand_dims(pred_alphas[0:num_objs], 1)
            detections_box_2d = \
                np.hstack([valid_boxes_2d, valid_pred_alphas, new_valid_scores, classes])
            pred_dict[constants.KEY_BOX_2D] = detections_box_2d

        return pred_dict

    def save_predictions(self, sample_name, predictions, sample_dict, output_dirs):

        predictions = self.format_predictions(self.output_types, predictions, sample_dict)

        if constants.KEY_INST_XYZ_MAP_LOCAL in self.output_types:
            # TODO: save global maps instead?
            pred_inst_xyz_map_local = predictions[constants.KEY_INST_XYZ_MAP_LOCAL]
            pred_inst_xyz_map_local_out_dir = output_dirs[constants.OUT_DIR_XYZ_MAP_LOCAL]
            pred_out_path = pred_inst_xyz_map_local_out_dir + '/{}.npy'.format(sample_name)
            np.save(pred_out_path, pred_inst_xyz_map_local.astype(np.float16))

        if constants.KEY_VALID_MASK_MAPS in self.output_types:
            # Save masks
            pred_valid_mask_maps = predictions[constants.KEY_VALID_MASK_MAPS].astype(np.uint8) * 255
            for mask_idx, mask in enumerate(pred_valid_mask_maps):
                cv2.imwrite(pred_inst_xyz_map_local_out_dir + '/{}_{}.png'.format(
                    sample_name, mask_idx), mask)

        if constants.KEY_CENTROIDS in self.output_types:
            # Save 3D box detections
            detections_box_3d = predictions[constants.KEY_BOX_3D]
            pred_box_3d_out_dir = output_dirs[constants.OUT_DIR_BOX_3D]
            pred_box_3d_path = pred_box_3d_out_dir + '/{}.txt'.format(sample_name)
            np.savetxt(pred_box_3d_path, detections_box_3d, fmt='%0.5f')

            # Save 2D box detections
            detections_box_2d = predictions[constants.KEY_BOX_2D]
            pred_box_2d_out_dir = output_dirs[constants.OUT_DIR_BOX_2D]
            pred_box_2d_path = pred_box_2d_out_dir + '/{}.txt'.format(sample_name)
            np.savetxt(pred_box_2d_path, detections_box_2d, fmt='%0.5f')

    def evaluate_predictions(self, prediction_dict, gt_dict):
        """Add nodes to evaluate metrics within the graph"""

        print('Adding metrics:')

        metrics_dict = {}
        metrics_debug_dict = {}

        if constants.KEY_INST_XYZ_MAP_LOCAL in self.output_types:
            print('\t{}'.format(constants.KEY_INST_XYZ_MAP_LOCAL))

            with tf.variable_scope('xyz_to_points'):
                # Parse predictions
                pred_inst_xyz_maps_local = prediction_dict[constants.KEY_INST_XYZ_MAP_LOCAL]

                gt_inst_xyz_maps_local = gt_dict[constants.KEY_INST_XYZ_MAP_LOCAL]
                gt_valid_mask_maps = gt_dict[constants.KEY_VALID_MASK_MAPS]

                # Multiply by valid mask
                valid_pred_inst_xyz_maps_local = pred_inst_xyz_maps_local * gt_valid_mask_maps
                valid_gt_inst_xyz_maps_local = gt_inst_xyz_maps_local * gt_valid_mask_maps

                # Reshape to (batch_size, n_points, 3)
                valid_pred_inst_points = tf.reshape(
                    valid_pred_inst_xyz_maps_local, (self.num_boxes, -1, 3))
                valid_gt_points = tf.reshape(
                    valid_gt_inst_xyz_maps_local, (self.num_boxes, -1, 3))

            # Calculate number of valid pixels per instance
            num_objs = self.pl_num_objs
            num_valid_pixels = tf.reduce_sum(gt_valid_mask_maps[0:num_objs], axis=[1, 2, 3])

            metrics_debug_dict.update({
                'num_objs': num_objs,
                'num_valid_pixels': num_valid_pixels,
            })

            # TODO: Move emd and chamfer into separate file
            # Calculate the average earth mover's distance over the number of boxes
            with tf.variable_scope('emd'):
                match = tf_approxmatch.approx_match(valid_pred_inst_points, valid_gt_points)
                all_distances = tf_approxmatch.match_cost(
                    valid_pred_inst_points, valid_gt_points, match)
                obj_distances = all_distances[0:num_objs]
                valid_distances = obj_distances / num_valid_pixels
                metrics_dict[constants.METRIC_EMD] = valid_distances

            metrics_debug_dict.update({
                'all_distances': all_distances,
                'obj_distances': obj_distances,
                'valid_distances': valid_distances,
            })

            # Calculate the average chamfer distance over the number of boxes
            with tf.variable_scope('chamfer_dist'):
                dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(
                    valid_pred_inst_points, valid_gt_points)
                all_chamfer_dists = tf.reduce_sum(dist1, axis=1) + tf.reduce_sum(dist2, axis=1)
                obj_chamfer_dists = all_chamfer_dists[0:num_objs]
                valid_chamfer_dists = obj_chamfer_dists / num_valid_pixels
                metrics_dict[constants.METRIC_CHAMFER] = valid_chamfer_dists

            metrics_debug_dict.update({
                'all_chamfer_dists': all_chamfer_dists,
                'obj_chamfer_dists': obj_distances,
                'valid_chamfer_dists': valid_chamfer_dists,
            })

        if constants.KEY_CENTROIDS in self.output_types:
            print('\t{}'.format(constants.KEY_CENTROIDS))

            with tf.variable_scope('centroid'):

                # Get proposal cen z
                prop_cen_z = prediction_dict[constants.KEY_PROP_CEN_Z][0:self.pl_num_objs]

                # Get predictions
                pred_cens = prediction_dict[constants.KEY_CENTROIDS][0:self.pl_num_objs]

                gt_cens = gt_dict[constants.KEY_CENTROIDS][0:self.pl_num_objs]

                # Calculate error
                prop_cen_z_err = gt_cens[:, 2:3] - prop_cen_z
                cen_errs = gt_cens - pred_cens

                metrics_dict[constants.METRIC_PROP_CEN_Z_ERR] = prop_cen_z_err
                metrics_dict[constants.METRIC_CEN_X_ERR] = cen_errs[:, 0]
                metrics_dict[constants.METRIC_CEN_Y_ERR] = cen_errs[:, 1]
                metrics_dict[constants.METRIC_CEN_Z_ERR] = cen_errs[:, 2]

        if constants.KEY_LWH in self.output_types:
            print('\t{}'.format(constants.KEY_LWH))

            with tf.variable_scope('dim_offs'):
                # Get predictions
                pred_dim_offs = prediction_dict[constants.KEY_LWH + '_offs'][0:self.pl_num_objs]

                gt_dim_offs = gt_dict[constants.KEY_LWH + '_offs'][0:self.pl_num_objs]

                # Calculate error
                dim_errs = gt_dim_offs - pred_dim_offs

                metrics_dict[constants.METRIC_DIM_ERR] = dim_errs

        if constants.KEY_VIEW_ANG in self.output_types:
            print('\t{}'.format(constants.KEY_VIEW_ANG))

            with tf.variable_scope('view_ang'):
                # Get predictions
                pred_view_ang = prediction_dict[constants.KEY_VIEW_ANG][0:self.pl_num_objs]

                gt_view_ang_3d = gt_dict[constants.KEY_VIEW_ANG][0:self.pl_num_objs]

                # Calculate error
                view_ang_errs = gt_view_ang_3d - pred_view_ang
                metrics_dict[constants.METRIC_VIEW_ANG_ERR] = view_ang_errs

        print('Done adding metrics')

        return metrics_dict, metrics_debug_dict

    def get_variable_restore_map(self,
                                 fine_tune_checkpoint_type='detection',
                                 load_all_detection_checkpoint_vars=False):
        """Returns a map of variables to load from a foreign checkpoint.

        See parent class for details.

        Args:
          fine_tune_checkpoint_type: whether to restore from a full detection
            checkpoint (with compatible variable names) or to restore from a
            classification checkpoint for initialization prior to training.
            Valid values: `detection`, `classification`. Default 'detection'.
          load_all_detection_checkpoint_vars: whether to load all variables (when
             `fine_tune_checkpoint_type` is `detection`). If False, only variables
             within the feature extractor scopes are included. Default False.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        Raises:
          ValueError: if fine_tune_checkpoint_type is neither `classification`
            nor `detection`.
        """
        if fine_tune_checkpoint_type not in ['detection', 'classification']:
            raise ValueError('Not supported fine_tune_checkpoint_type: {}'.format(
                fine_tune_checkpoint_type))

        variables_to_restore = tf.global_variables()
        variables_to_restore.append(slim.get_or_create_global_step())
        # Only load feature extractor variables to be consistent with loading from
        # a classification checkpoint.
        include_patterns = None
        if not load_all_detection_checkpoint_vars:
            include_patterns = [
                'FirstStageFeatureExtractor'
                'FirstStageFeatureExtractor_full',
                'FirstStageFeatureExtractor_crop',
                'SecondStageFeatureExtractor'
            ]
        feature_extractor_variables = tf.contrib.framework.filter_variables(
            variables_to_restore, include_patterns=include_patterns)
        return {var.op.name: var for var in feature_extractor_variables}
