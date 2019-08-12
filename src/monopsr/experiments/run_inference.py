import argparse
import os

import tensorflow as tf

import monopsr
from monopsr.builders.dataset_builder import DatasetBuilder
from monopsr.core import config_utils
from monopsr.core.evaluator import Evaluator
from monopsr.core.models.monopsr.monopsr_model import MonoPSRModel


def main(_):
    parser = argparse.ArgumentParser()

    # Example usage
    # --checkpoint_name='monopsr_model_000'
    # --data_split='test'
    # --ckpt_num='80000'
    # Optional arg:
    # --device=0

    default_checkpoint_name = 'monopsr_model_000'

    default_ckpt_num = 'all'
    default_data_split = 'val'
    default_det_2d_score_thr = [0.2, 0.2, 0.2]
    default_device = '0'

    parser.add_argument('--checkpoint_name',
                        type=str,
                        dest='checkpoint_name',
                        default=default_checkpoint_name,
                        help='Checkpoint name must be specified as a str.')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split,
                        help='Data split must be specified e.g. val or test')

    parser.add_argument('--ckpt_num',
                        nargs='+',
                        dest='ckpt_num',
                        default=default_ckpt_num,
                        help='Checkpoint number ex. 80000')

    parser.add_argument('--det_2d_score_thr',
                        type=int,
                        dest='det_2d_score_thr',
                        default=default_det_2d_score_thr,
                        help='2D detection score threshold.')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default=default_device,
                        help='CUDA device id')

    args = parser.parse_args()

    experiment_config = args.checkpoint_name + '.yaml'

    # Read the config from the experiment folder
    experiment_config_path = monopsr.data_dir() + '/outputs/' + \
                             args.checkpoint_name + '/' + experiment_config

    config = config_utils.parse_yaml_config(experiment_config_path)

    # Overwrite 2D detection score threshold
    config.dataset_config.mscnn_thr = args.det_2d_score_thr

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Run inference
    inference(config, args.data_split, args.ckpt_num)


def inference(config, data_split, ckpt_indices):

    train_val_test = 'test'

    dataset_config = config.dataset_config
    dataset_config.data_split = data_split
    if data_split == 'test':
        dataset_config.data_split_dir = 'testing'
        dataset_config.has_kitti_labels = False

    # Enable this to see the actually memory being used
    config.allow_gpu_mem_growth = True

    # Remove augmentation during evaluation in test mode
    dataset_config.aug_config.box_jitter_type = None

    # Build the dataset object
    dataset = DatasetBuilder.build_kitti_dataset(dataset_config,
                                                 train_val_test=train_val_test)

    # Setup the model
    model_name = config.model_name
    model_config = config.model_config

    with tf.Graph().as_default():

        if model_name == 'monopsr':
            model = MonoPSRModel(model_config,
                                 train_val_test=train_val_test,
                                 dataset=dataset)
        else:
            raise ValueError('Invalid model_name')

        if ckpt_indices == 'all':
            model_evaluator = Evaluator(model, config, eval_mode='test',
                                        skip_evaluated_checkpoints=True,
                                        do_kitti_native_eval=False)

            model_evaluator.repeated_checkpoint_run()
        else:
            model_evaluator = Evaluator(model, config, eval_mode='test',
                                        skip_evaluated_checkpoints=False,
                                        do_kitti_native_eval=False)
            model_evaluator.run_latest_checkpoints(ckpt_indices)


if __name__ == '__main__':
    tf.app.run()
