import sys

import numpy as np
import os

from monopsr.datasets.kitti.kitti_dataset import KittiDataset
from monopsr.utils import demo_utils


def main():
    """ Converts a set of network predictions into text files required for
    KITTI evaluation.
    """

    ##############################
    # Options
    ##############################
    checkpoint_name = 'monopsr_model_000'

    data_split = 'val'
    # data_split = 'test'

    global_steps = [100000]

    score_threshold = 0.1

    save_2d = False  # Save 2D predictions
    save_3d = True   # Save 2D and 3D predictions together

    # Checkpoints below this are skipped
    min_step = 20000

    ##############################
    # End of Options
    ##############################

    config, predictions_base_dir = demo_utils.get_experiment_info(checkpoint_name)

    # Overwrite defaults
    dataset_config = config.dataset_config
    dataset_config.data_split = data_split
    dataset_config.aug_list = []

    if data_split == 'test':
        dataset_config.data_split_dir = 'testing'

    dataset = KittiDataset(dataset_config, train_val_test='test')

    for step_idx in range(len(global_steps)):

        global_step = global_steps[step_idx]

        pred_box_2d_dir = predictions_base_dir + '/predictions_box_2d/{}/{}'.format(
            dataset.data_split, global_step)

        pred_box_3d_dir = predictions_base_dir + '/predictions_box_3d/{}/{}'.format(
            dataset.data_split, global_step)

        # Skip first checkpoint
        if int(global_step) < min_step:
            continue

        # 2D and 3D prediction directories
        kitti_predictions_2d_dir = predictions_base_dir + \
            '/kitti_predictions_2d/' + \
            dataset.data_split + '/' + \
            str(score_threshold) + '/' + \
            str(global_step) + '/data'
        kitti_predictions_3d_dir = predictions_base_dir + \
            '/kitti_predictions_3d/' + \
            dataset.data_split + '/' + \
            str(score_threshold) + '/' + \
            str(global_step) + '/data'

        if save_2d and not os.path.exists(kitti_predictions_2d_dir):
            os.makedirs(kitti_predictions_2d_dir)
        if save_3d and not os.path.exists(kitti_predictions_3d_dir):
            os.makedirs(kitti_predictions_3d_dir)

        # Do conversion
        num_samples = dataset.num_samples
        num_valid_samples = 0

        print('\nGlobal step:', global_step)

        if save_2d:
            print('2D Detections saved to:', kitti_predictions_2d_dir)
        if save_3d:
            print('3D Detections saved to:', kitti_predictions_3d_dir)

        for sample_idx in range(num_samples):

            # Print progress
            sys.stdout.write('\rConverting {} / {}'.format(
                sample_idx + 1, num_samples))
            sys.stdout.flush()

            sample_name = dataset.sample_list[sample_idx].name

            prediction_file = sample_name + '.txt'

            kitti_predictions_2d_file_path = kitti_predictions_2d_dir + \
                '/' + prediction_file
            kitti_predictions_3d_file_path = kitti_predictions_3d_dir + \
                '/' + prediction_file

            predictions_2d_file_path = pred_box_2d_dir + '/' + prediction_file
            predictions_3d_file_path = pred_box_3d_dir + '/' + prediction_file

            # If no predictions, skip to next file
            if not os.path.exists(predictions_3d_file_path):
                if save_2d:
                    np.savetxt(kitti_predictions_2d_file_path, [])
                if save_3d:
                    np.savetxt(kitti_predictions_3d_file_path, [])
                continue

            all_predictions_2d = np.loadtxt(predictions_2d_file_path).reshape(-1, 7)
            all_predictions_3d = np.loadtxt(predictions_3d_file_path).reshape(-1, 9)

            score_filter = all_predictions_3d[:, 7] >= score_threshold
            all_predictions_2d = all_predictions_2d[score_filter]
            all_predictions_3d = all_predictions_3d[score_filter]

            # If no predictions, skip to next file
            if len(all_predictions_3d) == 0:
                if save_2d:
                    np.savetxt(kitti_predictions_2d_file_path, [])
                if save_3d:
                    np.savetxt(kitti_predictions_3d_file_path, [])
                continue

            num_valid_samples += 1

            # To keep each value in its appropriate position, an array of zeros
            # (N, 16) is allocated but only values [4:16] are used
            kitti_predictions = np.zeros([len(all_predictions_3d), 16])

            # Get object types
            all_pred_classes = all_predictions_3d[:, 8].astype(np.int32)
            obj_types = [dataset.classes[class_idx]
                         for class_idx in all_pred_classes]

            # Truncation and Occlusion are always empty (see below)

            # Alpha
            kitti_predictions[:, 3] = all_predictions_2d[:, 4]

            # 2D predictions
            kitti_predictions[:, 4:8] = all_predictions_2d[:, [1, 0, 3, 2]]

            # 3D predictions
            # (l, w, h)
            kitti_predictions[:, 8] = all_predictions_3d[:, 5]
            kitti_predictions[:, 9] = all_predictions_3d[:, 4]
            kitti_predictions[:, 10] = all_predictions_3d[:, 3]
            # (x, y, z)
            kitti_predictions[:, 11:14] = all_predictions_3d[:, 0:3]
            # (ry, score)
            kitti_predictions[:, 14:16] = all_predictions_3d[:, 6:8]

            # Round detections to 3 decimal places
            kitti_predictions = np.round(kitti_predictions, 5)

            # Empty Truncation, Occlusion
            kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                         dtype=np.int32)
            # Empty 3D (x, y, z)
            kitti_empty_2 = -1 * np.ones((len(kitti_predictions), 3),
                                         dtype=np.int32)
            # Empty 3D (h, w, l)
            kitti_empty_3 = -1000 * np.ones((len(kitti_predictions), 3),
                                            dtype=np.int32)
            # Empty 3D (ry)
            kitti_empty_4 = -10 * np.ones((len(kitti_predictions), 1),
                                          dtype=np.int32)

            # Stack 2D predictions text
            kitti_text_2d = np.column_stack([obj_types,
                                             kitti_empty_1,
                                             kitti_predictions[:, 3:8],
                                             kitti_empty_2,
                                             kitti_empty_3,
                                             kitti_empty_4,
                                             kitti_predictions[:, 15]])

            # Stack 3D predictions text
            kitti_text_3d = np.column_stack([obj_types,
                                             kitti_empty_1,
                                             kitti_predictions[:, 3:16]])

            # Save to text files
            if save_2d:
                np.savetxt(kitti_predictions_2d_file_path, kitti_text_2d,
                           newline='\r\n', fmt='%s')
            if save_3d:
                np.savetxt(kitti_predictions_3d_file_path, kitti_text_3d,
                           newline='\r\n', fmt='%s')

        print('\nNum valid:', num_valid_samples)
        print('Num samples:', num_samples)


if __name__ == '__main__':
    main()
