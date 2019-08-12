from copy import deepcopy

import yaml

import monopsr
from monopsr.core import config_utils
from monopsr.datasets.kitti.kitti_dataset import KittiDataset


class DatasetBuilder:
    """Class with static methods to return preconfigured dataset objects
    """

    CONFIG_DEFAULTS_YAML = \
        """
        dataset_type: 'kitti_obj'
        
        use_mscnn_detections: True
        mscnn_thr: [0.2, 0.2, 0.2]
        
        batch_size: 1
        oversample: True
        
        num_boxes: 32
        num_alpha_bins: 8
        alpha_bin_overlap: 0.1
        centroid_type: 'middle'
        
        classes: ['Car']
        
        # Object Filtering
        obj_filter_config:
            # Note: Object types filtered based on classes
            difficulty_str: 'hard'
            occlusion: !!null
            truncation: 0.3
            box_2d_height: !!null
            depth_range: [5, 45]
        
        # Augmentation
        aug_config:
            use_image_aug: False
            box_jitter_type: 'oversample'  # 'oversample', 'all', !!null

        name: 'kitti'
        dataset_dir: '~/Kitti/object'
        data_split: 'train'
        data_split_dir: 'training'
        has_kitti_labels: True

        depth_version: 'multiscale'  # bilateral, multiscale, wavedata
        
        # depth_2_multiscale, depth_2_wavedata
        instance_version: 'depth_2_multiscale'
        
        """

    KITTI_TRAIN = 'kitti_obj_train'
    KITTI_VAL = 'kitti_obj_val'
    KITTI_TRAINVAL = 'kitti_obj_trainval'
    KITTI_TEST = 'kitti_obj_test'

    @staticmethod
    def get_config_obj(dataset_type):

        config_obj = config_utils.config_dict_to_object(
            yaml.load(DatasetBuilder.CONFIG_DEFAULTS_YAML))

        if dataset_type == DatasetBuilder.KITTI_TRAIN:
            return config_obj
        elif dataset_type == DatasetBuilder.KITTI_VAL:
            config_obj.data_split = 'val'
        elif dataset_type == DatasetBuilder.KITTI_TRAINVAL:
            config_obj.data_split = 'trainval'
        elif dataset_type == DatasetBuilder.KITTI_TEST:
            config_obj.data_split = 'test'
            config_obj.data_split_dir = 'testing'
            config_obj.has_kitti_labels = False
        else:
            raise ValueError('Invalid dataset type', dataset_type)

        return config_obj

    @staticmethod
    def build_kitti_dataset(dataset_config, train_val_test='train'):

        if isinstance(dataset_config, str):
            config_obj = DatasetBuilder.get_config_obj(dataset_config)
            return KittiDataset(config_obj, train_val_test)

        return KittiDataset(dataset_config, train_val_test)


if __name__ == '__main__':

    train_dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAIN)
    print(train_dataset)
