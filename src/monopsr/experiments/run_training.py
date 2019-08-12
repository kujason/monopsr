import argparse
import datetime
import filecmp
import os
import shutil

import tensorflow as tf

import monopsr
from monopsr.core.models.monopsr.monopsr_model import MonoPSRModel
from monopsr.core import trainer, config_utils
from monopsr.datasets.kitti.kitti_dataset import KittiDataset

tf.logging.set_verbosity(tf.logging.ERROR)


def main(_):
    parser = argparse.ArgumentParser()

    # Defaults
    default_config_path = monopsr.root_dir() + '/configs/monopsr_model_000.yaml'

    default_data_split = 'train'
    default_device = '1'

    parser.add_argument('--config_path',
                        type=str,
                        dest='config_path',
                        default=default_config_path,
                        help='Path to the config')

    parser.add_argument('--data_split',
                        type=str,
                        dest='data_split',
                        default=default_data_split,
                        help='Data split for training')

    parser.add_argument('--device',
                        type=str,
                        dest='device',
                        default=default_device,
                        help='CUDA device id')

    args = parser.parse_args()

    # Set CUDA device id
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    config_path = args.config_path
    config = config_utils.parse_yaml_config(config_path)

    # Copy config into outputs folder
    config_copy_path = config.exp_output_dir + '/{}.yaml'.format(config.config_name)
    if not os.path.exists(config_copy_path):
        shutil.copyfile(args.config_path, config_copy_path)
    else:
        # Check that config hasn't changed
        if not filecmp.cmp(config_path, config_copy_path):
            # Make a backup of the old config
            datetime_str = str(datetime.datetime.now())
            backup_copy_path = config_copy_path + '.' + datetime_str
            shutil.copyfile(config_copy_path, backup_copy_path)

            # Copy new config
            shutil.copyfile(args.config_path, config_copy_path)
            print('Config file has changed since ', datetime_str)

    # Overwrite data split
    dataset_config = config.dataset_config
    dataset_config.data_split = args.data_split

    train(config)


def train(config):

    dataset = KittiDataset(config.dataset_config, train_val_test='train')

    train_val_test = 'train'
    model_config = config.model_config
    model_name = config.model_name

    with tf.Graph().as_default():
        if model_name == 'monopsr':
            model = MonoPSRModel(model_config,
                                 train_val_test=train_val_test,
                                 dataset=dataset)
        else:
            raise ValueError('Invalid model_name')

        trainer.train(model, config)


if __name__ == '__main__':
    tf.app.run()
