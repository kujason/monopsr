import numpy as np

import tensorflow as tf

from object_detection.utils import variables_helper


def load_checkpoints(checkpoint_dir, saver):

    # Load latest checkpoint if available
    all_checkpoint_states = tf.train.get_checkpoint_state(checkpoint_dir)
    if all_checkpoint_states is not None:
        all_checkpoint_paths = all_checkpoint_states.all_model_checkpoint_paths
        # Save the checkpoint list into saver.last_checkpoints
        saver.recover_last_checkpoints(all_checkpoint_paths)
    else:
        all_checkpoint_paths = None
        print('No checkpoints found')

    return np.asarray(all_checkpoint_paths)
    # return saver.last_checkpoints


def get_global_step(sess, global_step_tensor):
    # Read the global step if restored
    global_step = tf.train.global_step(sess, global_step_tensor)
    return global_step


def split_checkpoint_step(checkpoint_dir):
    """Helper function to return the checkpoint index number.

    Args:
        checkpoint_dir: Path directory of the checkpoints

    Returns:
        checkpoint_id: An int representing the checkpoint index
    """

    checkpoint_name = checkpoint_dir.split('/')[-1]
    return int(checkpoint_name.split('-')[-1])


def restore_weights_by_scope(sess, pretrained_ckpt_path, scope):
    """Load in pre-trained weights for variables identified by a specific scope

    Args:
        sess: A TensorFlow session
        pretrained_ckpt_path: path to location of pre-trained checkpoint to load weights from
        scope: the variable scope that identifies the variables to load weights for
    """

    # Find the variables that belong to a particular scope
    variables_to_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    # Create saver for the variables
    pretrained_weight_saver = tf.train.Saver(variables_to_restore)

    # Use the saver to load in the checkpoint
    print('Loading in pre-trained {} weights'.format(scope))
    pretrained_weight_saver.restore(sess, pretrained_ckpt_path)


def restore_obj_detection_api_weights(sess, model, pretrained_ckpt_path):
    """Finds variables pertaining to the object detection api within the model and loads in the
    pre-trained weights for these variables

    Args:
        sess: A TensorFlow session
        model: detection model object
        pretrained_ckpt_path: path for the pre-trained weights
    """

    # Returns mapping of the variables used for 2D object detection
    var_map = model.get_variable_restore_map(
        fine_tune_checkpoint_type='detection',
        load_all_detection_checkpoint_vars=True)

    if model.net_type in ['resnet50_16x', 'resnet101_16x',
                          'resnet101_4x_separate_weights',
                          'resnet101_4x_squash', 'resnet101_4x_bigmon']:
        # Change to TensorFlow Object Detection API standard variables names
        full_var_map_cleaned = {}
        crop_var_map_cleaned = {}
        for key, var in var_map.items():
            # TODO: Remove hard coding of variable scopes
            full_var_map_cleaned[key.replace('FirstStageFeatureExtractor_full/',
                                             'FirstStageFeatureExtractor/')] = var
            crop_var_map_cleaned[key.replace('FirstStageFeatureExtractor_crop/',
                                             'FirstStageFeatureExtractor/')] = var

        # Find the variables in the checkpoint and match them to the ones in the var_maps
        available_full_var_map = (variables_helper.get_variables_available_in_checkpoint(
            full_var_map_cleaned, pretrained_ckpt_path,
            include_global_step=False))

        available_crop_var_map = (variables_helper.get_variables_available_in_checkpoint(
            crop_var_map_cleaned, pretrained_ckpt_path,
            include_global_step=False))

        # Load in weights
        full_init_saver = tf.train.Saver(available_full_var_map)
        full_init_saver.restore(sess, pretrained_ckpt_path)

        crop_init_saver = tf.train.Saver(available_crop_var_map)
        crop_init_saver.restore(sess, pretrained_ckpt_path)

    else:
        available_var_map = (variables_helper.get_variables_available_in_checkpoint(
            var_map, pretrained_ckpt_path,
            include_global_step=False))

        # Load in weights
        init_saver = tf.train.Saver(available_var_map)
        init_saver.restore(sess, pretrained_ckpt_path)

    print('Loading in Object Detection API pre-trained weights')
