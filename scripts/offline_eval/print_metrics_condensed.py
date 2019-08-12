import numpy as np

import monopsr
from monopsr.core import constants


def get_top_metrics_strings(data, all_metric_names, steps):

    top_metrics = {}
    top_metrics_with_step = {}
    for metric_name in all_metric_names:

        metric_values = np.abs(data[metric_name])

        # Get top metric
        top_idx = np.argmin(metric_values)
        top_step = steps.take(top_idx).astype(np.int32)
        top_value = metric_values.take(top_idx)

        # If the top step is 0, then ground truth was likely used
        if top_step == 0:
            value_as_str = None
            combined_value_step_str = None
        else:
            value_as_str = str(round(top_value, 3))
            combined_value_step_str = value_as_str + '     (' + str(top_step) + ')'

        top_metrics['metric_' + metric_name] = value_as_str
        top_metrics_with_step['metric_' + metric_name] = combined_value_step_str

    return top_metrics, top_metrics_with_step


def get_specific_metrics_strings(data, all_metric_names, steps, checkpoint):

    top_metrics = {}
    top_metrics_with_step = {}

    idx = np.argmax(steps == checkpoint)

    for metric_name in all_metric_names:

        metric_values = np.abs(data[metric_name])

        # Get top metric
        top_value = metric_values.take(idx)

        value_as_str = str(round(top_value, 3))
        combined_value_step_str = value_as_str + '     (' + str(checkpoint) + ')'

        top_metrics['metric_' + metric_name] = value_as_str
        top_metrics_with_step['metric_' + metric_name] = combined_value_step_str

    return top_metrics, top_metrics_with_step


def main():
    """Prints top metrics in a condensed format
    """

    ##############################
    # Options
    ##############################
    checkpoint_name = 'monopsr_model_000'

    data_split = 'val'

    get_specific_chkpt = True
    checkpoint = 110000

    ##############################

    # Get paths
    metrics_dir = monopsr.scripts_dir() + '/offline_eval/metrics/{}/{}'.format(
        checkpoint_name, data_split)
    metrics_avg_path = metrics_dir + '/metrics_avg_{}.csv'.format(data_split)
    metrics_std_path = metrics_dir + '/metrics_std_{}.csv'.format(data_split)
    metrics_avg_abs_path = metrics_dir + '/metrics_avg_abs_{}.csv'.format(data_split)

    # Parse csv
    avg_data = np.genfromtxt(metrics_avg_path, dtype=np.float32, delimiter=',',
                             names=True)
    std_data = np.genfromtxt(metrics_std_path, dtype=np.float32, delimiter=',',
                             names=True)
    avg_abs_data = np.genfromtxt(metrics_avg_abs_path, dtype=np.float32,
                                 delimiter=',', names=True)

    all_metric_names = avg_data.dtype.names

    # Checkpoint steps
    steps = avg_data['step']

    if get_specific_chkpt:
        top_avg, top_avg_with_step = get_specific_metrics_strings(avg_data, all_metric_names,
                                                                  steps, checkpoint)
        top_std, top_std_with_step = get_specific_metrics_strings(std_data, all_metric_names,
                                                                  steps, checkpoint)
        top_avg_abs, top_avg_abs_with_step = get_specific_metrics_strings(
            avg_abs_data, all_metric_names, steps, checkpoint)

    else:
        top_avg, top_avg_with_step = get_top_metrics_strings(avg_data, all_metric_names, steps)
        top_std, top_std_with_step = get_top_metrics_strings(std_data, all_metric_names, steps)
        top_avg_abs, top_avg_abs_with_step = get_top_metrics_strings(avg_abs_data, all_metric_names,
                                                                     steps)

    print('Top metrics:')
    print('{:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} '
          '{:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format(
              'MAE',
              'RMSE',
              'EMD',
              'CHAMFER',
              'ABS_CEN_Z_ERR',
              'STD_CEN_Z_ERR',
              'ABS_CEN_Y_ERR',
              'STD_CEN_Y_ERR',
              'ABS_CEN_X_ERR',
              'STD_CEN_X_ERR',
              'ABS_VIEW_ANG_ERR',
              'STD_VIEW_ANG_ERR',
              'ABS_LWH_ERR',
              'STD_LWH_ERR',
              'ABS_PROP_CEN_Z_ERR',
              'STD_PROP_CEN_Z_ERR'))

    # Metrics without step
    print('{:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'
          '{:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format(
              str(top_avg.get(constants.METRIC_MAE)),
              str(top_avg.get(constants.METRIC_RMSE)),
              str(top_avg.get(constants.METRIC_EMD)),
              str(top_avg.get(constants.METRIC_CHAMFER)),
              str(top_avg_abs.get(constants.METRIC_CEN_Z_ERR)),
              str(top_std.get(constants.METRIC_CEN_Z_ERR)),
              str(top_avg_abs.get(constants.METRIC_CEN_Y_ERR)),
              str(top_std.get(constants.METRIC_CEN_Y_ERR)),
              str(top_avg_abs.get(constants.METRIC_CEN_X_ERR)),
              str(top_std.get(constants.METRIC_CEN_X_ERR)),
              str(top_avg_abs.get(constants.METRIC_VIEW_ANG_ERR)),
              str(top_std.get(constants.METRIC_VIEW_ANG_ERR)),
              str(top_avg_abs.get(constants.METRIC_DIM_ERR)),
              str(top_std.get(constants.METRIC_DIM_ERR)),
              str(top_avg_abs.get(constants.METRIC_PROP_CEN_Z_ERR)),
              str(top_std.get(constants.METRIC_PROP_CEN_Z_ERR))))

    print('\nMetrics with step (for copying into spreadsheet):')

    # Metrics with step (to be copied to spreadsheet)
    # Semi colon is added for easy splitting in spreadsheets
    print('{:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s} '
          '{:>15s} {:>15s} {:>15s} {:>15s} {:>15s} {:>15s}'.format(
              str(top_avg_with_step.get(constants.METRIC_MAE)) + ';',
              str(top_avg_with_step.get(constants.METRIC_RMSE)) + ';',
              str(top_avg_with_step.get(constants.METRIC_EMD)) + ';',
              str(top_avg_with_step.get(constants.METRIC_CHAMFER)) + ';',
              str(top_avg_abs_with_step.get(constants.METRIC_CEN_Z_ERR)) + ';',
              str(top_std_with_step.get(constants.METRIC_CEN_Z_ERR)) + ';',
              str(top_avg_abs_with_step.get(constants.METRIC_CEN_Y_ERR)) + ';',
              str(top_std_with_step.get(constants.METRIC_CEN_Y_ERR)) + ';',
              str(top_avg_abs_with_step.get(constants.METRIC_CEN_X_ERR)) + ';',
              str(top_std_with_step.get(constants.METRIC_CEN_X_ERR)) + ';',
              str(top_avg_abs_with_step.get(constants.METRIC_VIEW_ANG_ERR)) + ';',
              str(top_std_with_step.get(constants.METRIC_VIEW_ANG_ERR)) + ';',
              str(top_avg_abs_with_step.get(constants.METRIC_DIM_ERR)) + ';',
              str(top_std_with_step.get(constants.METRIC_DIM_ERR)) + ';',
              str(top_avg_abs_with_step.get(constants.METRIC_PROP_CEN_Z_ERR)) + ';',
              str(top_std_with_step.get(constants.METRIC_PROP_CEN_Z_ERR))
          ))


if __name__ == '__main__':
    main()
