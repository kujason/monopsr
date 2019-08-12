import csv
import datetime
import os
import subprocess
import sys
from distutils import dir_util

import monopsr
import numpy as np
import tensorflow as tf
from PIL import Image
from monopsr.core import box_3d_projector
from monopsr.core import summary_utils
from monopsr.datasets.kitti import calib_utils


def save_predictions_box_2d_in_kitti_format(score_threshold,
                                            dataset,
                                            predictions_base_dir,
                                            predictions_box_2d_dir,
                                            global_step):
    """Converts and saves predictions (box_3d) into text files required for KITTI evaluation

    Args:
        score_threshold: score threshold to filter predictions
        dataset: Dataset object
        predictions_box_2d_dir: predictions (box_3d) folder
        predictions_base_dir: predictions base folder
        global_step: global step
    """

    score_threshold = round(score_threshold, 3)
    data_split = dataset.data_split

    # Output folder
    kitti_predictions_2d_dir = predictions_base_dir + \
        '/kitti_predictions_3d/{}/{}/{}/data'.format(data_split, score_threshold, global_step)
    if not os.path.exists(kitti_predictions_2d_dir):
        os.makedirs(kitti_predictions_2d_dir)

    # Do conversion
    num_samples = dataset.num_samples
    num_valid_samples = 0

    print('\nGlobal step:', global_step)
    print('Converting detections from:', predictions_box_2d_dir)
    print('3D Detections being saved to:', kitti_predictions_2d_dir)

    for sample_idx in range(num_samples):

        # Print progress
        sys.stdout.write('\rConverting {} / {}'.format(sample_idx + 1, num_samples))
        sys.stdout.flush()

        sample_name = dataset.sample_list[sample_idx].name

        prediction_file = sample_name + '.txt'
        kitti_predictions_2d_file_path = kitti_predictions_2d_dir + '/' + prediction_file
        predictions_file_path = predictions_box_2d_dir + '/' + prediction_file

        # If no predictions, skip to next file
        if not os.path.exists(predictions_file_path):
            np.savetxt(kitti_predictions_2d_file_path, [])
            continue

        all_predictions = np.loadtxt(predictions_file_path).reshape(-1, 6)

        # Change the order to be (x1, y1, x2, y2)
        copied_predictions = np.copy(all_predictions)
        all_predictions[:, 0:4] = copied_predictions[:, [1, 0, 3, 2]]

        score_filter = all_predictions[:, 4] >= score_threshold
        all_predictions = all_predictions[score_filter]

        # If no predictions, skip to next file
        if len(all_predictions) == 0:
            np.savetxt(kitti_predictions_2d_file_path, [])
            continue

        num_valid_samples += 1

        # To keep each value in its appropriate position, an array of -1000
        # (N, 16) is allocated but only values [4:16] are used
        kitti_predictions = np.full([all_predictions.shape[0], 16], -1000.0)

        # To avoid estimating alpha, -10 is used as a placeholder
        kitti_predictions[:, 3] = -10.0

        # Get object types
        all_pred_classes = all_predictions[:, 5].astype(np.int32)
        obj_types = [dataset.classes[class_idx] for class_idx in all_pred_classes]

        # 2D predictions
        kitti_predictions[:, 4:8] = all_predictions[:, 0:4]

        # Score
        kitti_predictions[:, 15] = all_predictions[:, 4]

        # Round detections to 3 decimal places
        kitti_predictions = np.round(kitti_predictions, 3)

        # Stack 3D predictions text
        kitti_text_3d = np.column_stack([obj_types,
                                         kitti_predictions[:, 1:16]])

        # Save to text files
        np.savetxt(kitti_predictions_2d_file_path, kitti_text_3d,
                   newline='\r\n', fmt='%s')

    print('\nNum valid:', num_valid_samples)
    print('Num samples:', num_samples)


def save_predictions_box_3d_in_kitti_format(score_threshold,
                                            dataset,
                                            predictions_base_dir,
                                            predictions_box_3d_dir,
                                            predictions_box_2d_dir,
                                            global_step,
                                            project_3d_box=False):
    """Converts and saves predictions (box_3d) into text files required for KITTI evaluation

    Args:
        score_threshold: score threshold to filter predictions
        dataset: Dataset object
        predictions_box_3d_dir: predictions (box_3d) folder
        predictions_box_2d_dir: predictions (box_2d) folder
        predictions_base_dir: predictions base folder
        global_step: global step
        project_3d_box: Bool whether to project 3D box to image space to get 2D box
    """

    score_threshold = round(score_threshold, 3)
    data_split = dataset.data_split

    # Output folder
    kitti_predictions_3d_dir = predictions_base_dir + \
        '/kitti_predictions_3d/{}/{}/{}/data'.format(data_split, score_threshold, global_step)
    if not os.path.exists(kitti_predictions_3d_dir):
        os.makedirs(kitti_predictions_3d_dir)

    # Do conversion
    num_samples = dataset.num_samples
    num_valid_samples = 0

    print('\nGlobal step:', global_step)
    print('Converting detections from:', predictions_box_3d_dir)
    print('3D Detections being saved to:', kitti_predictions_3d_dir)

    for sample_idx in range(num_samples):

        # Print progress
        sys.stdout.write('\rConverting {} / {}'.format(sample_idx + 1, num_samples))
        sys.stdout.flush()

        sample_name = dataset.sample_list[sample_idx].name

        prediction_file = sample_name + '.txt'
        kitti_predictions_3d_file_path = kitti_predictions_3d_dir + '/' + prediction_file
        predictions_3d_file_path = predictions_box_3d_dir + '/' + prediction_file
        predictions_2d_file_path = predictions_box_2d_dir + '/' + prediction_file

        # If no predictions, skip to next file
        if not os.path.exists(predictions_3d_file_path):
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        all_predictions_3d = np.loadtxt(predictions_3d_file_path)
        if len(all_predictions_3d) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        all_predictions_3d = all_predictions_3d.reshape(-1, 9)
        all_predictions_2d = np.loadtxt(predictions_2d_file_path).reshape(-1, 7)

        # # Swap l, w for predictions where w > l
        # swapped_indices = all_predictions[:, 4] > all_predictions[:, 3]
        # fixed_predictions = np.copy(all_predictions)
        # fixed_predictions[swapped_indices, 3] = all_predictions[
        #     swapped_indices, 4]
        # fixed_predictions[swapped_indices, 4] = all_predictions[
        #     swapped_indices, 3]

        score_filter = all_predictions_3d[:, 7] >= score_threshold
        all_predictions_3d = all_predictions_3d[score_filter]
        all_predictions_2d = all_predictions_2d[score_filter]

        # If no predictions, skip to next file
        if len(all_predictions_3d) == 0:
            np.savetxt(kitti_predictions_3d_file_path, [])
            continue

        # Project to image space
        sample_name = prediction_file.split('.')[0]

        # Load image for truncation
        image = Image.open(dataset.get_rgb_image_path(sample_name))
        # TODO: Check which camera
        cam_p = calib_utils.get_frame_calib(dataset.calib_dir, sample_name).p2

        if project_3d_box:
            boxes = []
            image_filter = []
            for i in range(len(all_predictions_3d)):
                box_3d = all_predictions_3d[i, 0:7]
                img_box = box_3d_projector.project_to_image_space(
                    box_3d, cam_p,
                    truncate=True, image_size=image.size)

                # Skip invalid boxes (outside image space)
                if img_box is None:
                    image_filter.append(False)
                    continue

                image_filter.append(True)
                boxes.append(img_box)

            boxes_2d = np.asarray(boxes)
            all_predictions_3d = all_predictions_3d[image_filter]
            all_predictions_2d = all_predictions_2d[image_filter]

        else:
            # Get 2D boxes from 2D predictions
            boxes_2d = all_predictions_2d[:, [1, 0, 3, 2]]

        # If no predictions, skip to next file
        if len(all_predictions_3d) == 0:
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
        kitti_predictions[:, 4:8] = boxes_2d

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
        kitti_predictions = np.round(kitti_predictions, 3)

        # Empty Truncation, Occlusion
        kitti_empty_1 = -1 * np.ones((len(kitti_predictions), 2),
                                     dtype=np.int32)

        # Stack 3D predictions text
        kitti_text_3d = np.column_stack([obj_types,
                                         kitti_empty_1,
                                         kitti_predictions[:, 3:16]])

        # Save to text files
        np.savetxt(kitti_predictions_3d_file_path, kitti_text_3d,
                   newline='\r\n', fmt='%s')

    print('\nNum valid:', num_valid_samples)
    print('Num samples:', num_samples)


def _add_metrics_csv_header(metric_names, csv_writer):
    # Remove 'metric_' prefix
    metric_header_names = []
    for metric_name in metric_names:
        if metric_name.startswith('metric'):
            metric_header_names.append(metric_name[7:])
        else:
            metric_header_names.append(metric_name)

    metric_headers = ['{}'.format(metric_header_name).rjust(12)
                      for metric_header_name in metric_header_names]
    csv_writer.writerow(['step'.rjust(8), *metric_headers])


def save_metrics(checkpoint_name, data_split, global_step,
                 metrics_dict, model_config, summary_writer):
    """Saves averages for metrics as a csv in <predictions/metrics/metrics.csv> and
    creates scalar summaries for tensorboard

    Args:
        checkpoint_name: Checkpoint name
        data_split: Data split
        global_step: Global step
        metrics_dict: Metrics dictionary
        model_config: Model config
        summary_writer: Summary writer object
    """

    # Create metrics dir
    metrics_dir = monopsr.scripts_dir() + '/offline_eval/metrics/{}/{}/'.format(
        checkpoint_name, data_split)
    os.makedirs(metrics_dir, exist_ok=True)

    # Setup file paths
    metrics_avg_path = os.path.join(metrics_dir, 'metrics_avg_{}.csv'.format(data_split))
    metrics_std_path = os.path.join(metrics_dir, 'metrics_std_{}.csv'.format(data_split))
    metrics_avg_abs_path = os.path.join(metrics_dir, 'metrics_avg_abs_{}.csv'.format(data_split))
    metrics_std_abs_path = os.path.join(metrics_dir, 'metrics_std_abs_{}.csv'.format(data_split))

    # Open files
    file_avg = open(metrics_avg_path, 'a')
    file_std = open(metrics_std_path, 'a')
    file_avg_abs = open(metrics_avg_abs_path, 'a')
    file_std_abs = open(metrics_std_abs_path, 'a')

    # Create csv writers
    csv_writer_avg = csv.writer(file_avg, delimiter=',')
    csv_writer_std = csv.writer(file_std, delimiter=',')
    csv_writer_avg_abs = csv.writer(file_avg_abs, delimiter=',')
    csv_writer_std_abs = csv.writer(file_std_abs, delimiter=',')

    metric_names = sorted(metrics_dict.keys())

    # Setup csv header if empty
    if os.stat(metrics_avg_path).st_size == 0:
        _add_metrics_csv_header(metric_names, csv_writer_avg)
    if os.stat(metrics_std_path).st_size == 0:
        _add_metrics_csv_header(metric_names, csv_writer_std)
    if os.stat(metrics_avg_abs_path).st_size == 0:
        _add_metrics_csv_header(metric_names, csv_writer_avg_abs)
    if os.stat(metrics_std_abs_path).st_size == 0:
        _add_metrics_csv_header(metric_names, csv_writer_std_abs)

    # Line to add to csv file
    global_step_str = '{}'.format(global_step).rjust(8)
    line_avg = [global_step_str]
    line_std = [global_step_str]
    line_avg_abs = [global_step_str]
    line_std_abs = [global_step_str]

    # Check which metrics to add to tensorboard
    metrics_to_show = np.asarray(model_config.metrics_to_show)

    for key in metric_names:

        values_for_metric = metrics_dict[key]

        # Calculate average and standard deviation
        avg_metric = np.mean(values_for_metric)
        std_metric = np.std(values_for_metric)

        # Calculate average and standard deviation of absolute values
        abs_values_for_metric = np.abs(values_for_metric)
        avg_abs_metric = np.mean(abs_values_for_metric)
        std_abs_metric = np.std(abs_values_for_metric)

        # Format and append metric value to line
        metric_avg_str = '{:.5f}'.format(avg_metric).rjust(12)
        metric_std_str = '{:.5f}'.format(std_metric).rjust(12)
        metric_avg_std_str = '{:.5f}'.format(avg_abs_metric).rjust(12)
        metric_std_std_str = '{:.5f}'.format(std_abs_metric).rjust(12)
        line_avg.append(metric_avg_str)
        line_std.append(metric_std_str)
        line_avg_abs.append(metric_avg_std_str)
        line_std_abs.append(metric_std_std_str)

        # Check if summary should be added to tensorboard
        config_indices = np.where(metrics_to_show[:, 0] == key)[0]
        if len(config_indices) > 0:
            for config_idx in config_indices:
                show_config = metrics_to_show[config_idx]
                show_metric_type = show_config[1]

                if show_metric_type == 'avg':
                    metric_to_show_value = avg_metric
                elif show_metric_type == 'std':
                    metric_to_show_value = std_metric
                elif show_metric_type == 'avg_abs':
                    metric_to_show_value = avg_abs_metric
                elif show_metric_type == 'std_abs':
                    metric_to_show_value = std_abs_metric
                else:
                    raise ValueError('Invalid show_metric_type', show_metric_type)

                # Add summary to tensorboard
                summary_utils.add_scalar_summary(
                    'metrics/{}/'.format(show_metric_type) + key,
                    metric_to_show_value, summary_writer, global_step)

    # Write lines to file
    csv_writer_avg.writerow(line_avg)
    csv_writer_std.writerow(line_std)
    csv_writer_avg_abs.writerow(line_avg_abs)
    csv_writer_std_abs.writerow(line_std_abs)


def set_up_summary_writer(train_config, data_split, sess):
    """Helper function to set up log directories and summary
    handlers.

    Args:
        train_config: training configuration
        sess: A tensorflow session
    """

    paths_config = train_config.paths_config

    logdir = paths_config.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    logdir = logdir + '/eval_{}/'.format(data_split)

    datetime_str = str(datetime.datetime.now())
    summary_writer = tf.summary.FileWriter(logdir + '/' + datetime_str, sess.graph)

    global_summaries = set([])
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    summary_merged = summary_utils.summaries_to_keep(summaries,
                                                     global_summaries,
                                                     histograms=False,
                                                     input_imgs=False,
                                                     input_bevs=False)

    return summary_writer, summary_merged


def print_inference_time_statistics(total_feed_dict_time,
                                    total_inference_time):

    # Print feed_dict time stats
    total_feed_dict_time = np.asarray(total_feed_dict_time)
    print('Feed dict time:')
    print('Min: ', np.round(np.min(total_feed_dict_time), 5))
    print('Max: ', np.round(np.max(total_feed_dict_time), 5))
    print('Mean: ', np.round(np.mean(total_feed_dict_time), 5))
    print('Median: ', np.round(np.median(total_feed_dict_time), 5))

    # Print inference time stats
    total_inference_time = np.asarray(total_inference_time)
    print('Inference time:')
    print('Min: ', np.round(np.min(total_inference_time), 5))
    print('Max: ', np.round(np.max(total_inference_time), 5))
    print('Mean: ', np.round(np.mean(total_inference_time), 5))
    print('Median: ', np.round(np.median(total_inference_time), 5))


def compile_kitti_native_code():
    """Compiles the kitti native code if not already compiled, or if it has been updated
    """

    kitti_native_code_dir = monopsr.top_dir() + '/scripts/offline_eval/kitti_native_eval'

    # Check if compiled already
    exists = os.path.exists(kitti_native_code_dir + '/evaluate_object_3d_offline')
    low_iou_exists = os.path.exists(kitti_native_code_dir + '/evaluate_object_3d_offline_low_iou')

    if not (exists and low_iou_exists):
        run_make_script = kitti_native_code_dir + '/run_make.sh'
        subprocess.call([run_make_script, kitti_native_code_dir])


def copy_kitti_native_code(checkpoint_name):
    """Copies and compiles kitti native code.

    It also creates necessary directories for storing the results
    of the kitti native evaluation code.
    """

    raise RuntimeError('Should not be used')

    monopsr_root_dir = monopsr.root_dir()
    kitti_native_code_copy = monopsr_root_dir + '/data/outputs/' + \
        checkpoint_name + '/predictions/kitti_native_eval/'

    # Only copy if the code has not been already copied over
    if not os.path.exists(kitti_native_code_copy):

        os.makedirs(kitti_native_code_copy)
        original_kitti_native_code = monopsr.top_dir() + \
            '/scripts/offline_eval/kitti_native_eval/'

        predictions_dir = monopsr_root_dir + '/data/outputs/' + \
            checkpoint_name + '/predictions/'
        # create dir for it first
        dir_util.copy_tree(original_kitti_native_code,
                           kitti_native_code_copy)
        # run the script to compile the c++ code
        script_folder = predictions_dir + \
            '/kitti_native_eval/'
        make_script = script_folder + 'run_make.sh'
        subprocess.call([make_script, script_folder])

    # Set up the results folders if they don't exist
    results_dir = monopsr.top_dir() + '/scripts/offline_eval/results'
    results_low_iou_dir = monopsr.top_dir() + '/scripts/offline_eval/results_low_iou'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(results_low_iou_dir):
        os.makedirs(results_low_iou_dir)


def run_kitti_native_script(checkpoint_name, data_split, kitti_score_threshold, global_step):
    """Runs the kitti native code script."""

    eval_script_dir = monopsr.top_dir() + '/scripts/offline_eval/kitti_native_eval'
    run_eval_script = eval_script_dir + '/run_eval.sh'
    kitti_predictions_dir = monopsr.data_dir() + \
        '/outputs/{}/predictions/kitti_predictions_3d/' \
        '{}/{}/{}'.format(checkpoint_name, data_split, kitti_score_threshold, global_step)
    results_dir = monopsr.top_dir() + '/scripts/offline_eval/results/{}'.format(data_split)
    os.makedirs(results_dir, exist_ok=True)

    # Round this because protobuf encodes default values as full decimal
    kitti_score_threshold = round(kitti_score_threshold, 3)

    subprocess.call([
        run_eval_script,
        str(eval_script_dir),
        str(checkpoint_name),
        str(kitti_score_threshold),
        str(global_step),
        str(kitti_predictions_dir),
        str(results_dir),
    ])


def run_kitti_native_script_with_low_iou(
        checkpoint_name, data_split, kitti_score_threshold, global_step):
    """Runs the low iou kitti native code script."""

    eval_script_dir = monopsr.top_dir() + '/scripts/offline_eval/kitti_native_eval'
    run_eval_script = eval_script_dir + '/run_eval_low_iou.sh'
    kitti_predictions_dir = monopsr.data_dir() + \
        '/outputs/{}/predictions/kitti_predictions_3d/' \
        '{}/{}/{}'.format(checkpoint_name, data_split, kitti_score_threshold, global_step)
    results_dir = monopsr.top_dir() + '/scripts/offline_eval/results_low_iou/{}'.format(data_split)
    os.makedirs(results_dir, exist_ok=True)

    # Round this because protobuf encodes default values as full decimal
    kitti_score_threshold = round(kitti_score_threshold, 3)

    subprocess.call([
        run_eval_script,
        str(eval_script_dir),
        str(checkpoint_name),
        str(kitti_score_threshold),
        str(global_step),
        str(kitti_predictions_dir),
        str(results_dir),
    ])
