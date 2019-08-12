import tensorflow as tf
from tensorflow.contrib import slim

from monopsr.core import constants
from monopsr.core.feature_extractors.faster_rcnn_resnet_v1_feature_extractor import \
    FasterRCNNResnet101FeatureExtractor


def get_net_config(model_config):

    net_type = model_config.net_type
    net_config = getattr(model_config.net_config, net_type)

    return net_config


def extract_features(model, net_type, model_config, input_dict, is_training):

    features_dict = {}

    net_config = model_config.net_config
    print('Net:', net_type)

    if net_type == 'resnet101_4x_squash':
        return build_resnet101_4x_squash(net_config, net_type, model, input_dict, features_dict)
    else:
        raise ValueError('Invalid net_type', net_type)


def build_resnet101_4x_squash(net_config, net_type, model, input_dict, features_dict):
    # Separate ResNet101s downsample the full and cropped image 4x,
    # features are squashed and max pooled,
    # then a simple decoder is used to obtain map features

    # Parse input dict
    crop_img = input_dict[constants.NET_IN_RGB_CROP]
    full_img = input_dict[constants.NET_IN_FULL_IMG]

    is_training = model.is_training

    # Separate ResNet101s for each input
    # TODO: Pass in extractor_config instead
    extractor_config = getattr(net_config, net_type)
    crop_img_encoder = FasterRCNNResnet101FeatureExtractor(
        is_training, extractor_config, output_stride=4)
    full_img_encoder = FasterRCNNResnet101FeatureExtractor(
        is_training, extractor_config, output_stride=4)

    crop_img_encoder_out, crop_img_end_points = crop_img_encoder.extract_proposal_features(
        crop_img, scope='FirstStageFeatureExtractor_crop')
    full_img_encoder_out, full_img_end_points = full_img_encoder.extract_proposal_features(
        full_img, scope='FirstStageFeatureExtractor_full')

    with tf.variable_scope('full_img_feature_crop'):
        # Crop and resize, then max pool the feature map from the full image
        full_img_feature_large_crop = tf.image.crop_and_resize(
            full_img_encoder_out, model.pl_boxes_2d_norm,
            tf.zeros(model.num_boxes, dtype=tf.int32),
            (model.map_roi_size[0] // 2, model.map_roi_size[1] // 2))
        full_img_feature_crop = slim.max_pool2d(full_img_feature_large_crop, [2, 2])

    # Concat the feature maps from the full and cropped images
    concat_features = tf.concat([crop_img_encoder_out, full_img_feature_crop], axis=3)

    # 1x1 conv and pool to reduce the dimensionality for fc layers
    with tf.variable_scope('squash'):
        features_squashed = slim.conv2d(concat_features, 512, [1, 1], scope='1x1_conv')
        features_pooled = slim.max_pool2d(features_squashed, [2, 2])

    with tf.variable_scope('map_decoder'):

        # Simple decoder to obtain map features
        resize1 = tf.image.resize_images(features_squashed,
                                         (model.map_roi_size[0] // 2, model.map_roi_size[1] // 2),
                                         align_corners=True)

        conv2 = slim.repeat(resize1, 2, slim.conv2d, 256, [3, 3],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            scope='conv2')
        resize2 = tf.image.resize_images(conv2,
                                         (model.map_roi_size[0], model.map_roi_size[1]),
                                         align_corners=True)

        conv3 = slim.repeat(resize2, 2, slim.conv2d, 128, [3, 3],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training},
                            scope='conv3')
        map_features = conv3

    features_dict.update({
        constants.FEATURES_FOR_MAP: map_features,
        constants.FEATURES_FOR_BOX_3D: features_pooled
    })

    return features_dict
