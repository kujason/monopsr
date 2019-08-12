import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from monopsr.core import constants, box_3d_projector, box_3d_encoder
from monopsr.datasets.kitti import obj_utils, instance_utils, depth_map_utils, calib_utils


class UniqueKeyDict:

    def __init__(self, init_entries=None):

        if init_entries is not None:
            self.dict = init_entries
        else:
            self.dict = {}

    def __getitem__(self, key):
        return self.dict[key]

    def add_unique_to_dict(self, new_entries):
        """Try to add entries to

        Args:
            new_entries:
        """
        for key, value in new_entries.items():
            # Check if key already exists
            existing_value = self.dict.get(key, None)
            if existing_value is None:
                self.dict.update({key: value})
            else:
                raise ValueError('Key {} already exists in output_dict'.format(key))


class MonoPSROutputBuilder:

    def __init__(self, output_config, model_config, dataset_config,
                 features_dict, num_boxes, map_roi_size, cam_p,
                 train_val_test):

        self.output_config = output_config.__dict__
        self.output_types = MonoPSROutputBuilder.get_output_types_list(output_config)

        self.model_config = model_config
        self.dataset_config = dataset_config

        # Create dictionaries
        self._output_dict = UniqueKeyDict()
        self._features_dict = UniqueKeyDict(features_dict)
        self._gt_dict = UniqueKeyDict()

        # Re-usable values
        self.num_boxes = num_boxes
        self.map_roi_size = map_roi_size
        self.cam_p = cam_p

        # Get expected set of feature maps
        self.features_for_map = features_dict[constants.FEATURES_FOR_MAP]
        self.features_for_box_3d = features_dict[constants.FEATURES_FOR_BOX_3D]

        self.train_val_test = train_val_test

        # Output mode
        if train_val_test == 'train':
            self.is_training = True
        else:
            self.is_training = False

        if train_val_test in ['train', 'val']:
            self.is_train_or_val = True
        else:
            self.is_train_or_val = False

    @staticmethod
    def get_output_types_list(output_config):
        output_types = sorted(
            [output_type
             for output_type in output_config.__dict__.keys()
             if not output_type.startswith('__')])
        return output_types

    def get_output_dict(self):
        return self._output_dict

    def get_output(self):
        return self._output_dict.dict

    def get_features_dict(self):
        return self._features_dict.dict

    def get_gt_dict(self):
        return self._gt_dict.dict

    def add_inst_xyz_maps_local(self, gt_inst_xyz_maps_local):
        output_key = constants.KEY_INST_XYZ_MAP_LOCAL
        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        with tf.variable_scope(output_key):
            output = slim.conv2d(
                self.features_for_map, 3, [3, 3],
                activation_fn=None, scope=output_key)
            self._output_dict.add_unique_to_dict({output_key: output})

            # Add ground truth
            if self.is_train_or_val:
                self._gt_dict.add_unique_to_dict({output_key: gt_inst_xyz_maps_local})

    def add_valid_mask_maps_output(self):
        output_key = constants.KEY_VALID_MASK_MAPS
        # output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, 'mask'))

        with tf.variable_scope(output_key):
            # TODO: output per class?
            output = slim.conv2d(
                self.features_for_map, 1, [3, 3],
                activation_fn=None, scope=output_key)
            self._output_dict.add_unique_to_dict({output_key: output})

    def get_proposal_fc_features(self):
        proposal_fc_features = self._features_dict[constants.FEATURES_PROPOSAL_FC_OUT]
        return proposal_fc_features

    def add_proposal_fc_features(self, boxes_2d, view_angs,
                                 class_indices, image_shape):
        """Add fully connected layers from the feature maps for box_3d

        Args:
            boxes_2d: Boxes 2D
            view_angs: viewing angle
            class_indices: class indices
            image_shape: image shape
        """

        with tf.variable_scope('proposal_fc'):

            # Get feature maps
            flat_img_features = slim.flatten(self.features_for_box_3d)

            with tf.variable_scope('normalize'):
                # Reshape box_2d info and initial estimates for feature concat
                box_2d_coords = obj_utils.tf_boxes_2d_ij_fmt(boxes_2d, self.cam_p)
                box_2d_heights = tf.expand_dims(boxes_2d[:, 2] - boxes_2d[:, 0], axis=1)

                # Normalize cen_y and box_2d_heights with mean values calculated in box_means.py
                box_2d_heights_norm = box_2d_heights / image_shape[0]

                # Normalize box coordinates by half the image shape
                half_img_height = image_shape[0] / 2.0
                half_img_width = image_shape[1] / 2.0
                box_2d_coords_norm = box_2d_coords / [half_img_height, half_img_width,
                                                      half_img_height, half_img_width]

                # Convert class index to one hot
                num_classes = len(self.dataset_config.classes)
                class_indices_one_hot = tf.one_hot(
                    tf.squeeze(class_indices), num_classes,
                    on_value=1.0, off_value=0.0)

            # Add fully connected layers
            with tf.variable_scope('proposal_fc'):

                # Add FC layer for image features
                img_fc = slim.fully_connected(flat_img_features, 1024, scope='img_fc')

                # Normalize camera matrix (make values between -1 and 1)
                cam_p_reshaped = tf.reshape(self.cam_p, [1, -1])
                cam_p_normalized = cam_p_reshaped / [1000.0, 1.0, 1000.0, 100.0,
                                                     1.0, 1000.0, 1000.0, 1.0,
                                                     1.0, 1.0, 1.0, 1.0]

                # Flatten and tile camera matrix
                cam_p_tiled = tf.tile(cam_p_normalized, [self.num_boxes, 1])

                # Concat normalized box_2d info
                features_concat = tf.concat(
                    [img_fc, box_2d_coords_norm, box_2d_heights_norm,
                     view_angs, class_indices_one_hot, cam_p_tiled], axis=1)

                # FC layers
                fc_layers_config = self.model_config.proposal_fc_layers
                fc_drop = features_concat
                for fc_idx, box_fc_size in enumerate(fc_layers_config.layer_sizes):
                    fc_out = slim.fully_connected(fc_drop, box_fc_size,
                                                  scope='fc{}'.format(fc_idx))
                    fc_drop = slim.dropout(fc_out, fc_layers_config.dropout_keep_prob,
                                           is_training=self.is_training,
                                           scope='fc{}_drop'.format(fc_idx))
                fc_out = fc_drop

            # Add to feature dict
            self._features_dict.add_unique_to_dict({constants.FEATURES_PROPOSAL_FC_OUT: fc_out})

    def get_regression_fc_features(self):
        features_dict = self.get_features_dict()
        return features_dict[constants.FEATURES_REGRESSION_FC_OUT]

    def add_regression_fc_features(self,
                                   boxes_2d,
                                   view_angs,
                                   class_indices,
                                   image_shape,
                                   est_lwh_off,
                                   est_alpha_bins,
                                   est_alpha_regs,
                                   prop_cen_y,
                                   prop_cen_z,
                                   max_depth):
        """Add fully connected layers from the feature maps for box_3d

        Args:
            boxes_2d: Boxes 2D
            view_angs: viewing angle
            class_indices: class indices
            image_shape: image shape
        """

        with tf.variable_scope('regression_fc'):

            # Get feature maps
            flat_img_features = slim.flatten(self.features_for_box_3d)

            # TODO: Combine with proposal fc features

            # Reshape box_2d info and initial estimates for feature concat
            box_2d_coords = obj_utils.tf_boxes_2d_ij_fmt(boxes_2d, self.cam_p)
            box_2d_heights = tf.expand_dims(boxes_2d[:, 2] - boxes_2d[:, 0], axis=1)

            # Normalize cen_y and box_2d_heights with mean values calculated in box_means.py
            box_2d_heights_norm = box_2d_heights / image_shape[0]

            # Normalize box coordinates by half the image shape
            half_img_height = image_shape[0] / 2.0
            half_img_width = image_shape[1] / 2.0
            box_2d_coords_norm = box_2d_coords / [half_img_height, half_img_width,
                                                  half_img_height, half_img_width]

            # Convert class index to one hot
            num_classes = len(self.dataset_config.classes)
            class_indices_one_hot = tf.one_hot(
                tf.squeeze(class_indices), num_classes,
                on_value=1.0, off_value=0.0)

            # Normalize y and z
            prop_cen_y_norm = prop_cen_y / 1.666754
            prop_cen_z_norm = prop_cen_z / max_depth

            # Add fully connected layers
            with tf.variable_scope('regression_fc'):
                # Add FC layer for image features
                img_fc = slim.fully_connected(flat_img_features, 1024, scope='img_fc')

                # Concat normalized box_2d info
                features_concat = tf.concat(
                    [img_fc, box_2d_coords_norm, box_2d_heights_norm,
                     view_angs, class_indices_one_hot,
                     est_lwh_off, est_alpha_bins, est_alpha_regs,
                     prop_cen_y_norm, prop_cen_z_norm], axis=1)

                # FC layers
                fc_layers_config = self.model_config.regression_fc_layers
                fc_drop = features_concat
                for fc_idx, box_fc_size in enumerate(fc_layers_config.layer_sizes):
                    fc_out = slim.fully_connected(fc_drop, box_fc_size,
                                                  scope='fc{}'.format(fc_idx))
                    fc_drop = slim.dropout(fc_out, fc_layers_config.dropout_keep_prob,
                                           is_training=self.is_training,
                                           scope='fc{}_drop'.format(fc_idx))
                fc_out = fc_drop

            # Add to feature dict
            self._features_dict.add_unique_to_dict({constants.FEATURES_REGRESSION_FC_OUT: fc_out})

    def add_alpha_output(self, features_to_use, gt_alpha, gt_alpha_dc):
        output_key = constants.KEY_ALPHA
        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        if output_type == 'dc':
            num_alpha_bins = self.dataset_config.num_alpha_bins
            orientation_outputs = slim.fully_connected(
                features_to_use,
                num_alpha_bins * 2,
                activation_fn=None,
                scope=output_key)
            pred_alpha_bins = orientation_outputs[:, 0:num_alpha_bins]
            pred_alpha_regs = \
                orientation_outputs[:, num_alpha_bins:num_alpha_bins * 2]

            self._output_dict.add_unique_to_dict({
                constants.KEY_ALPHA_BINS: pred_alpha_bins,
                constants.KEY_ALPHA_REGS: pred_alpha_regs,
            })

            # Add ground truth
            if self.is_train_or_val:
                self._gt_dict.add_unique_to_dict({
                    constants.KEY_ALPHA_BINS: gt_alpha_dc[0],
                    constants.KEY_ALPHA_REGS: gt_alpha_dc[1],
                })

        elif output_type == 'dc_rotation':

            num_alpha_bins = self.dataset_config.num_alpha_bins
            orientation_outputs = slim.fully_connected(
                features_to_use,
                num_alpha_bins * 3,
                activation_fn=None,
                scope=output_key)
            pred_alpha_bins = orientation_outputs[:, 0:num_alpha_bins]
            pred_alpha_regs_unnormalized = \
                orientation_outputs[:, num_alpha_bins:num_alpha_bins * 3]

            # Reshape for the sin and cos components
            pred_alpha_regs_unnormalized_reshape = tf.reshape(pred_alpha_regs_unnormalized,
                                                              (self.num_boxes, num_alpha_bins, 2))

            # Normalize
            pred_alpha_regs_normalized = tf.nn.l2_normalize(
                pred_alpha_regs_unnormalized_reshape, axis=2, epsilon=1e-12, name='L2-Norm-alpha')

            # Calculate estimated regression
            pred_alpha_regs = tf.atan2(pred_alpha_regs_normalized[:, :, 1],
                                       pred_alpha_regs_normalized[:, :, 0])

            self._output_dict.add_unique_to_dict({
                constants.KEY_ALPHA_BINS: pred_alpha_bins,
                constants.KEY_ALPHA_REGS: pred_alpha_regs,
            })

            # Add ground truth
            if self.is_train_or_val:
                self._gt_dict.add_unique_to_dict({
                    constants.KEY_ALPHA_BINS: gt_alpha_dc[0],
                    constants.KEY_ALPHA_REGS: gt_alpha_dc[1],
                })

        elif output_type == 'prob':
            num_alpha_bins = self.dataset_config.num_alpha_bins
            pred_alpha_bins = slim.fully_connected(
                features_to_use,
                num_alpha_bins,
                activation_fn=None,
                scope=output_key)

            # Use softmax to get probabilities
            pred_alpha_softmax = slim.softmax(pred_alpha_bins)

            # Calculate bin centres. Use NumPy as this will be the same each time
            angle_per_bin = 2 * np.pi / num_alpha_bins
            angle_per_half_bin = angle_per_bin / 2
            alpha_bin_centres = np.linspace(angle_per_half_bin, 2 * np.pi - angle_per_half_bin,
                                            num_alpha_bins)
            bin_centers_comp = tf.cast(tf.stack(
                (tf.cos(alpha_bin_centres), tf.sin(alpha_bin_centres)), axis=1), tf.float32)

            alpha_predicted_comp = tf.matmul(
                pred_alpha_softmax, bin_centers_comp)
            pred_alpha = tf.expand_dims(tf.atan2(
                alpha_predicted_comp[:, 1], alpha_predicted_comp[:, 0]), 1)

            self._output_dict.add_unique_to_dict({
                constants.KEY_ALPHA_BINS: pred_alpha_bins,
                constants.KEY_ALPHA: pred_alpha,
            })

            # Add ground truth
            if self.is_train_or_val:
                self._gt_dict.add_unique_to_dict({
                    constants.KEY_ALPHA_BINS: gt_alpha_dc[0],
                    constants.KEY_ALPHA: gt_alpha,
                })

        elif output_type == 'gt':
            pred_alpha_bins = gt_alpha_dc[0]
            pred_alpha_regs = gt_alpha_dc[1]

            self._output_dict.add_unique_to_dict({
                constants.KEY_ALPHA_BINS: pred_alpha_bins,
                constants.KEY_ALPHA_REGS: pred_alpha_regs,
            })

            # Add ground truth
            if self.is_train_or_val:
                self._gt_dict.add_unique_to_dict({
                    constants.KEY_ALPHA_BINS: gt_alpha_dc[0],
                    constants.KEY_ALPHA_REGS: gt_alpha_dc[1],
                })

        else:
            raise ValueError('Invalid output_type', output_type)

    def get_view_ang_output(self):
        return self._output_dict.dict[constants.KEY_VIEW_ANG]

    def get_cen_x_output(self):
        return self._output_dict.dict[constants.KEY_CEN_X]

    def get_cen_y_output(self):
        return self._output_dict.dict[constants.KEY_CEN_Y]

    def get_cen_z_output(self):
        return self._output_dict.dict[constants.KEY_CEN_Z]

    def get_prop_cen_z(self, boxes_2d, offset):

        with tf.variable_scope('prop_cen_z'):
            # Focal length
            cam_p = self.cam_p
            f = cam_p[0, 0]

            # Get estimated object height
            output_dict = self.get_output_dict()
            est_lwh = output_dict[constants.KEY_LWH]
            est_obj_h = est_lwh[:, 2]

            # Calculate 2D box height
            boxes_2d_h = boxes_2d[:, 2] - boxes_2d[:, 0]

            # Calculate initial estimate for z
            prop_cen_z = f * est_obj_h / boxes_2d_h + offset
            prop_cen_z = tf.expand_dims(prop_cen_z, axis=1)

            # Add centroid z estimate so it can be evaluated as a metric
            self._output_dict.add_unique_to_dict({
                constants.KEY_PROP_CEN_Z: prop_cen_z,
            })

        return prop_cen_z

    def get_prop_cen_y(self, boxes_2d, depth, class_name):

        prop_cen_y = instance_utils.tf_est_y_from_box_2d_and_depth(
            self.cam_p, boxes_2d, depth, class_name, trend_data='kitti')

        return prop_cen_y


    def add_cen_z_output(self, output_key, features_in, prop_cen_z, gt_cen_z):

        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        # Check input shapes
        assert prop_cen_z.shape[1] == 1
        # assert gt_cen_z.shape[1] == 1

        if output_type == 'offset':
            self._add_cen_z_offset(output_key, features_in, prop_cen_z, gt_cen_z)
        elif output_type == 'direct':
            self._add_cen_z_direct(output_key, features_in, gt_cen_z)
        else:
            raise ValueError('Invalid output_type', output_type)

    def _add_cen_z_offset(self, output_key, features_in, prop_cen_z, gt_cen_z):
        """Adds an output for predicting centroid z

        Args:
            output_key: Config key
            features_in: Features to use
            prop_cen_z: Initial centroid z estimate (for offset)
            gt_cen_z: Ground truth centroid z
        """

        with tf.variable_scope('cen_z_offs'):
            # Predict offsets from z estimate from the box_2d centre projection
            pred_cen_z_offsets = slim.fully_connected(
                features_in, 1,
                activation_fn=None,
                scope=output_key)

            pred_cen_z = prop_cen_z + pred_cen_z_offsets

            self._output_dict.add_unique_to_dict({
                output_key + '_offs': pred_cen_z_offsets,
                output_key: pred_cen_z,
            })

            # Add ground truth
            if self.is_train_or_val:
                gt_cen_z_offsets = gt_cen_z - prop_cen_z

                self._gt_dict.add_unique_to_dict({
                    output_key: gt_cen_z,
                    output_key + '_offs': gt_cen_z_offsets,
                })

    def _add_cen_z_direct(self, output_key, features_in, gt_cen_z):

        with tf.variable_scope('cen_z_direct'):
            # Predict z directly
            pred_cen_z = slim.fully_connected(
                features_in, 1,
                activation_fn=None,
                scope=output_key)

            self._output_dict.add_unique_to_dict({
                output_key: pred_cen_z,
            })

            # Add ground truth
            if self.is_train_or_val:
                self._gt_dict.add_unique_to_dict({
                    output_key: gt_cen_z,
                })

    def add_view_ang_output(self, output_key, features_in, est_view_angs, gt_view_angs):

        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        with tf.variable_scope(output_key):
            if output_type == 'offset':
                # Estimate offset from 2D viewing angle
                pred_view_ang_offsets = slim.fully_connected(
                    features_in, 1,
                    activation_fn=None,
                    scope=output_key)
                pred_view_angs = est_view_angs + pred_view_ang_offsets

            elif output_type == 'est':
                # Use 2D viewing angle as the estimate
                pred_view_angs = est_view_angs

                # No offset is predicted
                pred_view_ang_offsets = tf.constant(0.0)

            elif output_type == 'gt':
                # Use ground truth viewing angle
                pred_view_ang_offsets = gt_view_angs - est_view_angs
                pred_view_angs = gt_view_angs

            else:
                raise ValueError('Invalid output_type', output_type)

            self._output_dict.add_unique_to_dict({
                output_key + '_offs': pred_view_ang_offsets,
                output_key: pred_view_angs,
            })

            # Add ground truth
            if self.is_train_or_val:
                gt_view_ang_offsets = gt_view_angs - est_view_angs
                self._gt_dict.add_unique_to_dict({
                    output_key + '_offs': gt_view_ang_offsets,
                    output_key: gt_view_angs,
                })

    def add_cen_x_output(self, output_key, pred_cen_z, pred_view_angs):

        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        with tf.variable_scope(output_key):
            if output_type == 'from_view_ang_and_z':
                # Predict centroid x using viewing angle
                cam2_pred_cen_x = pred_cen_z * tf.tan(pred_view_angs)

                # Adjust for x_offset
                cam_p = self.cam_p
                x_offset = -cam_p[0, 3] / cam_p[0, 0]
                pred_cen_x = cam2_pred_cen_x + x_offset

            else:
                raise ValueError('Invalid output_type', output_type)

            self._output_dict.add_unique_to_dict({
                output_key: pred_cen_x,
            })

    def add_cen_y_output(self, output_key, features_in, prop_cen_y, gt_cen_y):
        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        with tf.variable_scope(output_key):
            if output_type == 'offset':
                # Estimate offset from estimated y
                pred_cen_y_offsets = slim.fully_connected(
                    features_in, 1,
                    activation_fn=None,
                    scope=output_key)

                pred_cen_y = prop_cen_y + pred_cen_y_offsets

            elif output_type == 'est':
                pred_cen_y = prop_cen_y

            elif output_type == 'gt':
                pred_cen_y_offsets = gt_cen_y - prop_cen_y
                pred_cen_y = gt_cen_y

            else:
                raise ValueError('Invalid output_type', output_type)

            self._output_dict.add_unique_to_dict({
                output_key + '_offs': pred_cen_y_offsets,
                output_key: pred_cen_y,
            })

            # Add ground truth
            if self.is_train_or_val:
                gt_cen_y_offsets = gt_cen_y - prop_cen_y

                self._gt_dict.add_unique_to_dict({
                    output_key + '_offs': gt_cen_y_offsets,
                    output_key: gt_cen_y,
                })

    def add_centroids_output(self, output_key, pred_cen_x, pred_cen_y, pred_cen_z, gt_centroids):

        with tf.variable_scope('centroid'):
            # Combine x, y, z predictions to get centroid
            pred_centroids = tf.concat([pred_cen_x, pred_cen_y, pred_cen_z], axis=1)
            self._output_dict.add_unique_to_dict({
                output_key: pred_centroids,
            })

        if self.is_train_or_val:
            self._gt_dict.add_unique_to_dict({
                output_key: gt_centroids,
            })

    def add_lwh_output(self, features_to_use, est_lwh, gt_lwh):

        output_key = constants.KEY_LWH
        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        with tf.variable_scope(output_key):
            if output_type == 'offset':
                pred_dim_offsets = slim.fully_connected(
                    features_to_use, 3,
                    activation_fn=None,
                    scope=output_key)
                pred_lwh = est_lwh + pred_dim_offsets
            elif output_type == 'est':
                # Use initial estimate
                pred_dim_offsets = est_lwh
                pred_lwh = est_lwh
            elif output_type == 'gt':
                # Use ground truth
                pred_dim_offsets = gt_lwh - est_lwh
                pred_lwh = gt_lwh
            else:
                raise ValueError('Invalid output_type', output_type)

            self._output_dict.add_unique_to_dict({
                output_key + '_offs': pred_dim_offsets,
                output_key: pred_lwh,
            })

            # Add ground truth
            if self.is_train_or_val:
                gt_pred_dim_offsets = gt_lwh - pred_lwh
                self._gt_dict.add_unique_to_dict({
                    output_key: gt_lwh,
                    output_key + '_offs': gt_pred_dim_offsets,
                })
        return pred_lwh

    def get_inst_xyz_map_global(self, pred_inst_xyz_maps_local,
                                pred_view_angs, pred_centroids):
        """Add global point cloud from local xyz map output

        Args:
            pred_inst_xyz_maps_local: Predicted local instance xyz maps
            pred_view_angs: Predicted viewing angles
            pred_centroids: Predicted centroids
        """
        output_key = constants.KEY_INST_XYZ_MAP_GLOBAL
        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        pred_inst_xyz_map_global = instance_utils.tf_inst_xyz_map_local_to_global(
            pred_inst_xyz_maps_local, self.map_roi_size, pred_view_angs, pred_centroids)

        return pred_inst_xyz_map_global

    def get_proj_err_maps_norm(self, pred_inst_xyz_map_global, pred_boxes_2d, valid_mask_maps):
        """Calculate normalized projection error

        Args:
            pred_inst_xyz_map_global:
            pred_boxes_2d:
            valid_mask_maps:

        Returns:
            proj_err_norm:
            proj_err_debug_dict:
        """

        num_boxes = pred_boxes_2d.shape[0]

        # Convert to padded point cloud
        pred_inst_pc_global = tf.reshape(
            tf.transpose(pred_inst_xyz_map_global, [0, 3, 1, 2]),
            [num_boxes, 3, -1])

        # Expected projection uv
        exp_proj_uv_map = instance_utils.tf_get_exp_proj_uv_map(pred_boxes_2d, self.map_roi_size)

        # Projection uv from prediction
        proj_uv_map_list = calib_utils.tf_project_pc_to_image(
            pred_inst_pc_global, self.cam_p, num_boxes)
        proj_uv_map = tf.reshape(
            tf.transpose(proj_uv_map_list, [0, 2, 1]),
            [num_boxes, *self.map_roi_size, 2])

        # Projection error
        proj_err_map = exp_proj_uv_map - proj_uv_map

        # Normalize projection error by size of 2D box
        boxes_h = pred_boxes_2d[:, 2] - pred_boxes_2d[:, 0]
        boxes_w = pred_boxes_2d[:, 3] - pred_boxes_2d[:, 1]
        boxes_wh = tf.stack([boxes_w, boxes_h], axis=1)
        proj_err_maps_norm = proj_err_map / tf.reshape(boxes_wh, (-1, 1, 1, 2))

        # Mask with valid pixel map and clip values
        proj_err_maps_norm *= valid_mask_maps
        proj_err_maps_norm = tf.clip_by_value(proj_err_maps_norm, -2.0, 2.0)

        # Normalize by number of pixels
        num_valid_pixels = tf.reduce_sum(valid_mask_maps, reduction_indices=[1, 2, 3])
        num_valid_pixels = tf.where(tf.less(num_valid_pixels, 1.0),
                                    tf.ones_like(num_valid_pixels), num_valid_pixels)

        proj_err_norm_sum = tf.reduce_sum(proj_err_maps_norm, reduction_indices=[1, 2, 3])
        proj_err_norm = proj_err_norm_sum / num_valid_pixels

        proj_err_debug_dict = {
            'pred_inst_pc_global': pred_inst_pc_global,
            'exp_proj_uv_map': exp_proj_uv_map,
            'pred_inst_xyz_map_global': pred_inst_xyz_map_global,

            'proj_uv_map_list': proj_uv_map_list,
            'proj_uv_map': proj_uv_map * valid_mask_maps,
            'proj_err_map': proj_err_map,

            'proj_err_map_norm': proj_err_maps_norm,
            'boxes_wh': boxes_wh,
            'cam_p_batch': tf.tile(self.cam_p, [32, 1]),
        }

        return proj_err_norm, proj_err_debug_dict

    def add_inst_depth_maps_global(self,
                                   pred_inst_depth_maps_local,
                                   gt_inst_depth_maps_global, rotate_view, box_2d):

        output_key = constants.KEY_INST_DEPTH_MAP_GLOBAL
        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        with tf.variable_scope(output_key):
            # Calculate global depth for instance
            pred_cen_z = self._output_dict[constants.KEY_CEN_Z]
            inst_view_ang = self._output_dict[constants.KEY_VIEW_ANG]

            pred_inst_depth_maps_global = instance_utils.tf_inst_depth_map_local_to_global(
                pred_inst_depth_maps_local, pred_cen_z, box_2d, inst_view_ang,
                self.map_roi_size, self.cam_p, rotate_view)

            self._output_dict.add_unique_to_dict({
                output_key: pred_inst_depth_maps_global,
            })

            if self.is_train_or_val:
                self._gt_dict.add_unique_to_dict({
                    output_key: gt_inst_depth_maps_global,
                })

    def add_inst_xyz_maps_global_from_depth(self, pred_inst_depth_map_global, boxes_2d,
                                            gt_inst_xyz_maps_global):
        output_key = constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH
        output_type = self.output_config[output_key]
        print('\t{:30s}{}'.format(output_key, output_type))

        with tf.variable_scope(output_key):
            # Convert to point cloud
            pred_inst_xyz_map_global_from_depth = np.asarray([
                tf.reshape(
                    depth_map_utils.tf_depth_patch_to_pc_map(
                        pred_inst_depth_map_global[box_idx, :, :, :],
                        boxes_2d[box_idx],
                        self.cam_p,
                        self.map_roi_size),
                    (-1, self.map_roi_size[0], self.map_roi_size[1], 3))
                for box_idx in np.arange(0, self.num_boxes, dtype=np.int32)])

            pred_inst_xyz_map_global_from_depth = tf.concat(
                list(pred_inst_xyz_map_global_from_depth), axis=0)

            self._output_dict.add_unique_to_dict({
                constants.KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH: pred_inst_xyz_map_global_from_depth,
            })

            if self.is_train_or_val:
                self._gt_dict.add_unique_to_dict({
                    output_key: gt_inst_xyz_maps_global,
                })


def score_boxes(dataset, sample_name, img_shape, boxes_2d, boxes_3d, valid_scores, max_depth=45.0):
    """Score 3D boxes based on 2D classification, depth, and fit between
    projected 3D box and the 2D detection

    Args:
        dataset: Dataset object
        sample_name: Sample name, e.g. '000050'
        img_shape: Image shape [h, w]
        boxes_2d: List of 2D boxes
        boxes_3d: List of 3D boxes
        valid_scores: List of box scores
        max_depth: Maximum depth, default 45m (95% of KITTI objects)
    """

    all_new_scores = np.zeros_like(valid_scores)
    for pred_idx, (box_2d, box_3d) in enumerate(zip(boxes_2d, boxes_3d)):

        # Project 3D box to 2D [x1, y1, x2, y2]
        cam_p = calib_utils.get_frame_calib(dataset.calib_dir, sample_name).p2

        projected_box_3d = box_3d_projector.project_to_image_space(
            box_3d, cam_p,
            truncate=True, image_size=(img_shape[1], img_shape[0]))

        # Change box_2d to iou format
        box_2d_iou_fmt = np.squeeze(box_3d_encoder.boxes_2d_to_iou_fmt([box_2d]))

        if projected_box_3d is None:
            # Truncated box
            new_score_box_fit = 0.1

        else:
            # Calculate corner error
            height = box_2d_iou_fmt[3] - box_2d_iou_fmt[1]
            width = box_2d_iou_fmt[2] - box_2d_iou_fmt[0]

            x1_err = np.abs((box_2d_iou_fmt[0] - projected_box_3d[0]) / width)
            x2_err = np.abs((box_2d_iou_fmt[2] - projected_box_3d[2]) / width)

            y1_err = np.abs((box_2d_iou_fmt[1] - projected_box_3d[1]) / height)
            y2_err = np.abs((box_2d_iou_fmt[3] - projected_box_3d[3]) / height)

            corner_err = x1_err + x2_err + y1_err + y2_err

            new_score_box_fit = 1.0 - corner_err

        depth = box_3d[2]
        new_score_depth = np.clip(1.0 - (depth / max_depth), 0.1, 1.0)

        new_score_depth_box_fit = (new_score_depth + new_score_box_fit) / 2.0

        mscnn_score = valid_scores[pred_idx]
        new_score = 0.95 * mscnn_score + 0.05 * new_score_depth_box_fit
        all_new_scores[pred_idx] = new_score

    return all_new_scores
