import time

import cv2
import numpy as np
from monopsr.builders.dataset_builder import DatasetBuilder

from monopsr.datasets.kitti import instance_utils, obj_utils
from monopsr.visualization import vis_utils


def main():
    ##############################
    # Options
    ##############################

    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAINVAL)

    sample_name = '000050'

    ##############################

    print('Showing instance masks for sample:', sample_name)

    # Load image and instances
    image = obj_utils.get_image(sample_name, dataset.image_2_dir)
    instance_img = cv2.imread(dataset.instance_dir + '/{}.png'.format(sample_name))
    instance_img = instance_img[:, :, 0].astype(np.uint8)

    # Get instance masks
    start = time.time()
    instance_masks = instance_utils.get_instance_mask_list(instance_img)
    print("Time to parse instance image:", str(time.time() - start))

    ##########################################################
    # Visualization
    ##########################################################

    # Shown on screen
    image_disp_size = (900, 300)
    vis_utils.cv2_imshow('Left Image', image,
                         size_wh=image_disp_size, location_xy=(80, 0))

    # Generate random colours for each instance
    for i in range(instance_masks.shape[0]):
        left_im_instances = instance_masks[i].astype(np.uint8) * 255

        vis_utils.cv2_imshow('Instances', left_im_instances,
                             size_wh=image_disp_size, location_xy=(80, 320))

        cv2.waitKey()


if __name__ == '__main__':
    main()
