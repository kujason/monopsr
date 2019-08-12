# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Resnet V1 Faster R-CNN implementation.

See "Deep Residual Learning for Image Recognition" by He et al., 2015.
https://arxiv.org/abs/1512.03385

Note: this implementation assumes that the classification checkpoint used
to finetune this model is trained using the same configuration as that of
the MSRA provided checkpoints
(see https://github.com/KaimingHe/deep-residual-networks), e.g., with
same preprocessing, batch norm scaling, etc.
"""
import tensorflow as tf

from object_detection.nets import resnet_utils
from object_detection.nets import resnet_v1

slim = tf.contrib.slim


class FasterRCNNFeatureExtractor(object):
    """Faster R-CNN Feature Extractor definition."""

    def __init__(self,
                 is_training,
                 first_stage_features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0,

                 output_stride=None):
        """Constructor.

        Args:
          is_training: A boolean indicating whether the training version of the
            computation graph should be constructed.
          first_stage_features_stride: Output stride of extracted RPN feature map.
          batch_norm_trainable: Whether to update batch norm parameters during
            training or not. When training with a relative large batch size
            (e.g. 8), it could be desirable to enable batch norm update.
          reuse_weights: Whether to reuse variables. Default is None.
          weight_decay: float weight decay for feature extractor (default: 0.0).
          output_stride: If None, then the output will be computed at the nominal
            network stride. If output_stride is not None, it specifies the requested
            ratio of input to output spatial resolution.
        """
        self._is_training = is_training
        self._first_stage_features_stride = first_stage_features_stride
        self._train_batch_norm = (batch_norm_trainable and is_training)
        self._reuse_weights = reuse_weights
        self._weight_decay = weight_decay
        self._output_stride = output_stride

    def preprocess(self, resized_inputs):
        """Feature-extractor specific preprocessing (minus image resizing)."""
        pass

    def extract_proposal_features(self, preprocessed_inputs, scope):
        """Extracts first stage RPN features.

        This function is responsible for extracting feature maps from preprocessed
        images.  These features are used by the region proposal network (RPN) to
        predict proposals.

        Args:
          preprocessed_inputs: A [batch, height, width, channels] float tensor
            representing a batch of images.
          scope: A scope name.

        Returns:
          rpn_feature_map: A tensor with shape [batch, height, width, depth]
          activations: A dictionary mapping activation tensor names to tensors.
        """
        with tf.variable_scope(scope, values=[preprocessed_inputs]):
            return self._extract_proposal_features(preprocessed_inputs, scope)

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        """Extracts first stage RPN features, to be overridden."""
        pass

    def extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features.

        Args:
          proposal_feature_maps: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
            representing the feature map cropped to each proposal.
          scope: A scope name.

        Returns:
          proposal_classifier_features: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, height, width, depth]
            representing box classifier features for each proposal.
        """
        with tf.variable_scope(
                scope, values=[proposal_feature_maps], reuse=tf.AUTO_REUSE):
            return self._extract_box_classifier_features(proposal_feature_maps, scope)

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features, to be overridden."""
        pass

    def restore_from_classification_checkpoint_fn(
            self,
            first_stage_feature_extractor_scope,
            second_stage_feature_extractor_scope):
        """Returns a map of variables to load from a foreign checkpoint.

        Args:
          first_stage_feature_extractor_scope: A scope name for the first stage
            feature extractor.
          second_stage_feature_extractor_scope: A scope name for the second stage
            feature extractor.

        Returns:
          A dict mapping variable names (to load from a checkpoint) to variables in
          the model graph.
        """
        variables_to_restore = {}
        for variable in tf.global_variables():
            for scope_name in [first_stage_feature_extractor_scope,
                               second_stage_feature_extractor_scope]:
                if variable.op.name.startswith(scope_name):
                    var_name = variable.op.name.replace(scope_name + '/', '')
                    variables_to_restore[var_name] = variable
        return variables_to_restore


class FasterRCNNResnetV1FeatureExtractor(FasterRCNNFeatureExtractor):
    """Faster R-CNN Resnet V1 feature extractor implementation."""

    def __init__(self,
                 architecture,
                 resnet_model,
                 is_training,
                 first_stage_features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0,

                 output_stride=None):
        """Constructor.

        Args:
          architecture: Architecture name of the Resnet V1 model.
          resnet_model: Definition of the Resnet V1 model.
          is_training: See base class.
          first_stage_features_stride: See base class.
          batch_norm_trainable: See base class.
          reuse_weights: See base class.
          weight_decay: See base class.
          output_stride: See base class.

        Raises:
          ValueError: If `first_stage_features_stride` is not 8 or 16.
        """
        if first_stage_features_stride != 8 and first_stage_features_stride != 16:
            raise ValueError('`first_stage_features_stride` must be 8 or 16.')
        self._architecture = architecture
        self._resnet_model = resnet_model
        super(FasterRCNNResnetV1FeatureExtractor, self).__init__(
            is_training, first_stage_features_stride, batch_norm_trainable,
            reuse_weights, weight_decay, output_stride)

    def preprocess(self, resized_inputs):
        """Faster R-CNN Resnet V1 preprocessing.

        VGG style channel mean subtraction as described here:
        https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

        Args:
          resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
            representing a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: A [batch, height_out, width_out, channels] float32
            tensor representing a batch of images.

        """
        channel_means = [123.68, 116.779, 103.939]
        return resized_inputs - [[channel_means]]

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        """Extracts first stage RPN features.

        Args:
          preprocessed_inputs: A [batch, height, width, channels] float32 tensor
            representing a batch of images.
          scope: A scope name.

        Returns:
          rpn_feature_map: A tensor with shape [batch, height, width, depth]
          activations: A dictionary mapping feature extractor tensor names to
            tensors

        Raises:
          InvalidArgumentError: If the spatial size of `preprocessed_inputs`
            (height or width) is less than 33.
          ValueError: If the created network is missing the required activation.
        """
        if len(preprocessed_inputs.get_shape().as_list()) != 4:
            raise ValueError('`preprocessed_inputs` must be 4 dimensional, got a '
                             'tensor of shape %s' % preprocessed_inputs.get_shape())
        shape_assert = tf.Assert(
            tf.logical_and(
                tf.greater_equal(tf.shape(preprocessed_inputs)[1], 33),
                tf.greater_equal(tf.shape(preprocessed_inputs)[2], 33)),
            ['image size must at least be 33 in both height and width.'])

        with tf.control_dependencies([shape_assert]):
            # Disables batchnorm for fine-tuning with smaller batch sizes.
            # TODO(chensun): Figure out if it is needed when image
            # batch size is bigger.
            with slim.arg_scope(
                resnet_utils.resnet_arg_scope(
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True,
                    weight_decay=self._weight_decay)):
                with tf.variable_scope(
                        self._architecture, reuse=self._reuse_weights) as var_scope:
                    _, activations = self._resnet_model(
                        preprocessed_inputs,
                        num_classes=None,
                        is_training=self._train_batch_norm,
                        global_pool=False,
                        output_stride=self._output_stride,
                        spatial_squeeze=False,
                        scope=var_scope)

        handle = scope + '/%s/block3' % self._architecture
        return activations[handle], activations

    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features.

        Args:
          proposal_feature_maps: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
            representing the feature map cropped to each proposal.
          scope: A scope name (unused).

        Returns:
          proposal_classifier_features: A 4-D float tensor with shape
            [batch_size * self.max_num_proposals, height, width, depth]
            representing box classifier features for each proposal.
        """
        with tf.variable_scope(self._architecture, reuse=self._reuse_weights):
            with slim.arg_scope(
                resnet_utils.resnet_arg_scope(
                    batch_norm_epsilon=1e-5,
                    batch_norm_scale=True,
                    weight_decay=self._weight_decay)):
                with slim.arg_scope([slim.batch_norm],
                                    is_training=self._train_batch_norm):
                    blocks = [
                        resnet_utils.Block('block4', resnet_v1.bottleneck, [{
                            'depth': 2048,
                            'depth_bottleneck': 512,
                            'stride': 1
                        }] * 3)
                    ]
                    proposal_classifier_features = resnet_utils.stack_blocks_dense(
                        proposal_feature_maps, blocks)
        return proposal_classifier_features


class FasterRCNNResnet50FeatureExtractor(FasterRCNNResnetV1FeatureExtractor):
    """Faster R-CNN Resnet 50 feature extractor implementation."""

    def __init__(self,
                 is_training,
                 config,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0,

                 output_stride=None):
        """Constructor.

        Args:
          is_training: See base class.
          config: Config object.
          batch_norm_trainable: See base class.
          reuse_weights: See base class.
          weight_decay: See base class.
          output_stride: See base class.

        Raises:
          ValueError: If `first_stage_features_stride` is not 8 or 16,
            or if `architecture` is not supported.
        """
        first_stage_features_stride = config.first_stage_features_stride
        super(FasterRCNNResnet50FeatureExtractor, self).__init__(
            'resnet_v1_50', resnet_v1.resnet_v1_50, is_training,
            first_stage_features_stride, batch_norm_trainable,
            reuse_weights, weight_decay, output_stride)


class FasterRCNNResnet101FeatureExtractor(FasterRCNNResnetV1FeatureExtractor):
    """Faster R-CNN Resnet 101 feature extractor implementation."""

    def __init__(self,
                 is_training,
                 config,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0,

                 output_stride=None):
        """Constructor.

        Args:
          is_training: See base class.
          config: Config object.
          batch_norm_trainable: See base class.
          reuse_weights: See base class.
          weight_decay: See base class.
          output_stride: See base class.

        Raises:
          ValueError: If `first_stage_features_stride` is not 8 or 16,
            or if `architecture` is not supported.
        """
        first_stage_features_stride = config.first_stage_features_stride
        super(FasterRCNNResnet101FeatureExtractor, self).__init__(
            'resnet_v1_101', resnet_v1.resnet_v1_101, is_training,
            first_stage_features_stride, batch_norm_trainable,
            reuse_weights, weight_decay, output_stride)


class FasterRCNNResnet152FeatureExtractor(FasterRCNNResnetV1FeatureExtractor):
    """Faster R-CNN Resnet 152 feature extractor implementation."""

    def __init__(self,
                 is_training,
                 first_stage_features_stride,
                 batch_norm_trainable=False,
                 reuse_weights=None,
                 weight_decay=0.0,

                 output_stride=None):
        """Constructor.

        Args:
          is_training: See base class.
          first_stage_features_stride: See base class.
          batch_norm_trainable: See base class.
          reuse_weights: See base class.
          weight_decay: See base class.
          output_stride: See base class.

        Raises:
          ValueError: If `first_stage_features_stride` is not 8 or 16,
            or if `architecture` is not supported.
        """
        super(FasterRCNNResnet152FeatureExtractor, self).__init__(
            'resnet_v1_152', resnet_v1.resnet_v1_152, is_training,
            first_stage_features_stride, batch_norm_trainable,
            reuse_weights, weight_decay, output_stride)
