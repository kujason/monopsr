import matplotlib.pyplot as plt
import numpy as np

import monopsr


def plot_values(data_type, data, checkpoint_name,
                top_n_to_print, inlier_percentiles, metric_name_filter):
    all_metric_names = data.dtype.names

    # Plot results
    num_metrics = len(all_metric_names) - 1

    plot_cols = num_metrics
    plot_rows = 2

    fig, ax_arr = plt.subplots(plot_rows, plot_cols,
                               figsize=(5.0 * plot_cols, 4. * plot_rows),
                               gridspec_kw={'height_ratios': [1, 2]})
    fig.canvas.set_window_title(data_type + ': ' + checkpoint_name)

    # Checkpoint steps
    steps = data['step']

    # Create plots
    for plot_idx in range(num_metrics):

        # Get metric values
        metric_name = all_metric_names[plot_idx + 1]

        if metric_name in metric_name_filter:
            continue

        metric_values = np.abs(data[metric_name])

        # Get top n indices
        top_n_indices = np.argsort(metric_values)[:top_n_to_print]

        top_n_steps = steps.take(top_n_indices).astype(np.int32)
        top_n_values = metric_values.take(top_n_indices)

        steps_formatted = ''.join([str(step).rjust(12) for step in top_n_steps])
        values_formatted = ''.join([str(value).rjust(12) for value in top_n_values])
        print('step  {:12s}'.format(metric_name), steps_formatted)
        print('value {:12s}'.format(metric_name), values_formatted)

        # Plot
        plot_row = 0
        plot_col = plot_idx
        ax_arr[plot_row, plot_col].plot(steps, metric_values)
        ax_arr[plot_row, plot_col].set_title(metric_name)
        plt.xticks(rotation=30)

        # Calculate outliers and replot in second row
        min_val, max_val = np.percentile(metric_values, inlier_percentiles)
        inlier_mask = (metric_values >= min_val) & (metric_values <= max_val)

        # Mask inliers
        inlier_steps = steps[inlier_mask]
        inlier_metric_values = metric_values[inlier_mask]

        # Plot with outliers removed
        plot_row = 1
        plot_col = plot_idx
        ax_arr[plot_row, plot_col].plot(inlier_steps, inlier_metric_values)
        ax_arr[plot_row, plot_col].set_title(metric_name + ' (inliers)')

        for tick in ax_arr[plot_row, plot_col].get_xticklabels():
            tick.set_rotation(20)


def main():
    """Plots metrics and prints top n checkpoints for each metric
    """

    ##############################
    # Options
    ##############################
    checkpoint_name = 'monopsr_model_000'

    data_split = 'val'

    # Top n indices to print
    top_n_to_print = 3

    # Metrics to ignore
    metric_name_filter = ['cen_x_err', 'cen_y_err']

    # Percentiles of values to plot
    inlier_percentiles = [0.0, 95.0]
    ##############################

    # Get paths
    metrics_dir = monopsr.scripts_dir() + '/offline_eval/metrics/{}/{}'.format(
        checkpoint_name, data_split)
    metrics_avg_path = metrics_dir + '/metrics_avg_{}.csv'.format(data_split)
    metrics_std_path = metrics_dir + '/metrics_std_{}.csv'.format(data_split)
    metrics_avg_abs_path = metrics_dir + '/metrics_avg_abs_{}.csv'.format(data_split)
    metrics_std_abs_path = metrics_dir + '/metrics_std_abs_{}.csv'.format(data_split)

    # Parse csv
    avg_data = np.genfromtxt(metrics_avg_path, dtype=np.float32, delimiter=',', names=True)
    std_data = np.genfromtxt(metrics_std_path, dtype=np.float32, delimiter=',', names=True)
    avg_abs_data = np.genfromtxt(metrics_avg_abs_path, dtype=np.float32, delimiter=',', names=True)
    std_abs_data = np.genfromtxt(metrics_std_abs_path, dtype=np.float32, delimiter=',', names=True)

    print('-----')
    print('Average:')
    plot_values('avg', avg_data, checkpoint_name,
                top_n_to_print, inlier_percentiles, metric_name_filter)

    print('-----')
    print('Standard Deviation')
    plot_values('std', std_data, checkpoint_name,
                top_n_to_print, inlier_percentiles, metric_name_filter)

    print('-----')
    print('Average (abs):')
    plot_values('avg_abs', avg_abs_data, checkpoint_name,
                top_n_to_print, inlier_percentiles, metric_name_filter)

    print('-----')
    print('Standard Deviation (abs)')
    plot_values('std_abs', std_abs_data, checkpoint_name,
                top_n_to_print, inlier_percentiles, metric_name_filter)

    plt.show(block=True)


if __name__ == '__main__':
    main()
