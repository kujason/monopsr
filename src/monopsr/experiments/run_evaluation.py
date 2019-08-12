import argparse
import os

import tensorflow as tf

import monopsr
from monopsr.core import config_utils
from monopsr.core.models.monopsr.monopsr_model import MonoPSRModel
from monopsr.core.evaluator import Evaluator
from monopsr.datasets.kitti.kitti_dataset import KittiDataset


def main(_):
    parser = argparse.ArgumentParser()

    default_config_path = monopsr.root_dir() + '/configs/monopsr_model_000.yaml'

    default_data_split = 'val'
    default_device = '0'

    parser.add_argument('--config_path',
                        type=str,
                        dest='config_path',
                        default=default_config_path,
                        help='Path to the pipeline config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split,
                        help='Data split for evaluation')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default=default_device,
                        help='CUDA device id')

    args = parser.parse_args()

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Parse yaml config
    config = config_utils.parse_yaml_config(args.config_path)

    # Overwrite data split
    dataset_config = config.dataset_config
    dataset_config.data_split = args.data_split

    evaluate(config)


def evaluate(config):

    # # Parse eval config
    # eval_mode = eval_config.eval_mode
    # if eval_mode not in ['val', 'test']:
    #     raise ValueError('Evaluation mode can only be set to `val` or `test`')
    # evaluate_repeatedly = eval_config.evaluate_repeatedly
    evaluate_repeatedly = True

    # Parse dataset config
    dataset_config = config.dataset_config
    data_split = dataset_config.data_split

    if data_split == 'test':
        dataset_config.data_split_dir = 'testing'
        dataset_config.has_kitti_labels = False
    else:
        dataset_config.data_split_dir = 'training'
        dataset_config.has_kitti_labels = True

    # Remove augmentation during evaluation
    dataset_config.aug_list = []

    # Build the dataset object
    dataset = KittiDataset(dataset_config, train_val_test='val')

    # Setup the model
    model_config = config.model_config
    model_name = config.model_name

    with tf.Graph().as_default():

        if model_name == 'monopsr':
            model = MonoPSRModel(model_config,
                                 train_val_test='val',
                                 dataset=dataset)

        else:
            raise ValueError('Invalid model name {}'.format(model_name))

        model_evaluator = Evaluator(model, config, eval_mode='val')

        if evaluate_repeatedly:
            model_evaluator.repeated_checkpoint_run()
        else:
            model_evaluator.run_latest_checkpoints()


if __name__ == '__main__':
    tf.app.run()
