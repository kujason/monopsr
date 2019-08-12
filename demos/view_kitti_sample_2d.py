import os

import matplotlib.pyplot as plt

from monopsr.datasets.kitti import obj_utils, calib_utils
from monopsr.visualization import vis_utils


def main():

    # Paths
    kitti_dir = os.path.expanduser('~/Kitti/object/')
    data_split_dir = 'training'

    image_dir = os.path.join(kitti_dir, data_split_dir) + '/image_2'
    label_dir = os.path.join(kitti_dir, data_split_dir) + '/label_2'
    calib_dir = os.path.join(kitti_dir, data_split_dir) + '/calib'

    sample_name = '000050'

    frame_calib = calib_utils.get_frame_calib(calib_dir, sample_name)
    cam_p = frame_calib.p2

    f, axes = vis_utils.plots_from_sample_name(image_dir, sample_name, 2, 1)

    # Load labels
    obj_labels = obj_utils.read_labels(label_dir, sample_name)
    for obj in obj_labels:

        # Draw 2D and 3D boxes
        vis_utils.draw_obj_as_box_2d(axes[0], obj)
        vis_utils.draw_obj_as_box_3d(axes[1], obj, cam_p)

    plt.show(block=True)


if __name__ == '__main__':
    main()
