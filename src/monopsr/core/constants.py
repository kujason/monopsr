
SAMPLE_IMAGE_INPUT = 'sample_image_input'

SAMPLE_NUM_OBJS = 'sample_num_objs'
SAMPLE_LABEL_BOXES_2D = 'sample_label_boxes_2d'
SAMPLE_LABEL_BOXES_2D_NORM = 'sample_label_boxes_2d_norm'
SAMPLE_LABEL_BOXES_3D = 'sample_label_boxes_3d'
SAMPLE_INSTANCE_MASKS = 'sample_instance_masks'
SAMPLE_ALPHAS = 'sample_alphas'
SAMPLE_ALPHA_BINS = 'sample_alpha_bins'
SAMPLE_ALPHA_REGS = 'sample_alpha_regressions'
SAMPLE_ALPHA_VALID_BINS = 'sample_alpha_valid_bins'

SAMPLE_PROP_CEN_Z_OFFSET = 'sample_prop_cen_z_offset'
SAMPLE_CEN_Z_EST = 'sample_cen_z_est'
SAMPLE_CEN_Y_EST = 'sample_cen_y_est'

SAMPLE_VIEWING_ANGLES_2D = 'sample_viewing_angles_2d'
SAMPLE_VIEWING_ANGLES_3D = 'sample_viewing_angles_3d'
SAMPLE_LABEL_CLASS_STRS = 'sample_label_class_strs'
SAMPLE_LABEL_CLASS_INDICES = 'sample_label_class_indices'
SAMPLE_LABEL_SCORES = 'sample_label_scores'

SAMPLE_DEPTH_MAP = 'sample_depth_map'
SAMPLE_XYZ_MAP = 'sample_xyz_map'

SAMPLE_CAM_P = 'sample_cam_p'

SAMPLE_NAME = 'sample_name'
SAMPLE_AUGS = 'sample_augs'

SAMPLE_MEAN_LWH = 'sample_mean_lwh'

# # # Shared Keys # # #
KEY_VALID_MASK_MAPS = 'valid_mask_maps'

# Local instance maps
KEY_INST_XYZ_MAP_LOCAL = 'inst_xyz_map_local'

# Global instance maps
KEY_INST_XYZ_MAP_GLOBAL = 'inst_xyz_map_global'
KEY_INST_PROJ_ERR_MAP = 'inst_proj_err_map'
KEY_INST_DEPTH_MAP_GLOBAL = 'inst_depth_map_global'
KEY_INST_XYZ_MAP_GLOBAL_FROM_DEPTH = 'inst_xyz_map_global_from_depth'

KEY_BOX_2D = 'box_2d'
KEY_BOX_3D = 'box_3d'

KEY_PROP_CEN_Z = 'prop_cen_z'

KEY_VIEW_ANG = 'view_ang'
KEY_CEN_X = 'cen_x'
KEY_CEN_Y = 'cen_y'
KEY_CEN_Z = 'cen_z'
KEY_CEN_Z_DC = 'cen_z_dc'

KEY_EST_CEN_Z = 'est_cen_z'
KEY_EST_CEN_Y = 'est_cen_y'

KEY_LWH = 'lwh'
KEY_ALPHA = 'alpha'
KEY_ALPHA_BINS = 'alpha_bins'
KEY_ALPHA_REGS = 'alpha_regs'

KEY_CENTROIDS = 'centroids'

# Net inputs
NET_IN_RGB_CROP = 'net_in_rgb_crop'
NET_IN_FULL_IMG = 'net_in_full_img'

# Net features
FEATURES_FOR_MAP = 'features_for_map'
FEATURES_FOR_BOX_3D = 'features_for_box_3d'
FEATURES_BOX_3D_FC_OUT = 'features_box_3d_fc_out'

FEATURES_PROPOSAL_FC_OUT = 'features_proposal_fc_out'
FEATURES_REGRESSION_FC_OUT = 'features_regression_fc_out'

# Output directions
OUT_DIR_BOX_2D = 'output_box_2d_dir'
OUT_DIR_BOX_3D = 'output_box_3d_dir'
OUT_DIR_XYZ_MAP_LOCAL = 'output_xyz_map_dir'
OUT_DIR_MASKS = 'output_masks_dir'
OUT_DIR_PROPS = 'output_proposal_dir'

# Metrics
METRIC_EMD = 'metric_emd'
METRIC_CHAMFER = 'metric_chamfer'
METRIC_RMSE = 'metric_rmse'
METRIC_MAE = 'metric_mae'

METRIC_VIEW_ANG_ERR = 'metric_view_ang_error'
METRIC_PROP_CEN_Z_ERR = 'metric_prop_cen_z_err'
METRIC_CEN_X_ERR = 'metric_cen_x_err'
METRIC_CEN_Y_ERR = 'metric_cen_y_err'
METRIC_CEN_Z_ERR = 'metric_cen_z_err'
METRIC_DIM_ERR = 'metric_dim_err'

# Centroid types
CENTROID_BOTTOM = 'bottom'
CENTROID_MIDDLE = 'middle'
