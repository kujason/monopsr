import matplotlib.pyplot as plt

import numpy as np

from monopsr.visualization import vis_utils


def parse_results_file(results_file):

    with open(results_file) as f:
        lines = f.readlines()

    num_lines = len(lines)
    line_idx = 0

    ap_dict = {}
    current_step = 0

    while line_idx < num_lines - 1:
        line = lines[line_idx].rstrip('\n')

        # Step
        if line.isdigit():
            current_step = int(line)
        else:
            ap_line = line.split(' ')

            if '_detection' in ap_line[0] or '_heading' in ap_line[0]:
                detection_type = str(ap_line[0])
                ap_vals = np.hstack([current_step, ap_line[2:]])

                if ap_dict.get(detection_type):
                    ap_dict[detection_type].append(ap_vals)
                else:
                    ap_dict.update({detection_type: [ap_vals]})
            else:
                # Ignore line (e.g. 'done', 'directory exists', etc.)
                pass

        line_idx += 1

    return ap_dict


def show_results(ap_dict, results_file, top_n_to_print):

    # Plot results (2D, 3D, BEV, 3D_heading, BEV_heading)
    num_ap_plots = len(ap_dict)
    plot_cols = 5
    plot_rows = int(np.ceil(num_ap_plots / plot_cols))

    fig, ax_arr = plt.subplots(plot_rows, plot_cols,
                               figsize=(17, 4 * plot_rows))
    fig.canvas.set_window_title(results_file)
    ax_arr = ax_arr.reshape(-1, plot_cols)

    print('-----')
    print(results_file)

    # Sliding window size
    window_size = 8
    half_window_size = window_size // 2

    # Create plots
    sorted_items = sorted(ap_dict.items())
    for plot_idx in range(num_ap_plots):
        # Get values from dict
        values = sorted_items[plot_idx]
        detection_type = values[0]
        lines = np.asarray(values[1], dtype=np.float32)
        steps = lines[:, 0]
        ap_values = lines[:, 1:]

        # Calculate sliding window average on moderate
        avg_mask = np.ones(window_size) / window_size
        ap_avg = np.convolve(ap_values[:, 1], avg_mask, mode='same')

        # Print top n indices
        top_n_med_indices = np.argsort(ap_values[:, 1])[-top_n_to_print:][::-1]
        print('{:25s}'.format(detection_type), steps.take(top_n_med_indices))

        # Plot
        plot_row = int(plot_idx / plot_cols)
        plot_col = plot_idx % plot_cols
        ax_arr[plot_row, plot_col].plot(steps, ap_values)
        ax_arr[plot_row, plot_col].plot(steps[half_window_size:-half_window_size],
                                        ap_avg[half_window_size:-half_window_size])
        ax_arr[plot_row, plot_col].set_title(detection_type)

    plt.legend(labels=['easy', 'medium', 'hard'])


def main():
    """Plots AP scores from the native eval script and prints top 5 checkpoints
        for each metric
    """

    checkpoint_name = 'monopsr_model_000'

    data_split = 'val'       # inf

    # Output from native eval
    results_file = 'results/{}/{}_results_0.1.txt'.format(data_split, checkpoint_name)
    results_low_iou_file = 'results_low_iou/{}/{}_results_low_iou_0.1.txt'.format(
        data_split, checkpoint_name)

    # Top n medium score indices to print
    top_n_to_print = 5

    ap_dict = parse_results_file(results_file)
    ap_low_iou_dict = parse_results_file(results_low_iou_file)

    show_results(ap_dict, results_file, top_n_to_print)
    show_results(ap_low_iou_dict, results_low_iou_file, top_n_to_print)

    for fig_idx, fig_num in enumerate(plt.get_fignums()):
        vis_utils.move_plt_figure(plt.figure(fig_num), 70, 30 + fig_idx * 500)

    plt.show(block=True)


if __name__ == '__main__':
    main()
