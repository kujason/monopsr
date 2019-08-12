import cv2
import matplotlib.pyplot as plt

from monopsr.builders.dataset_builder import DatasetBuilder
from monopsr.datasets.kitti import kitti_aug, obj_utils
from monopsr.visualization import vis_utils


def main():
    ##############################
    # Options
    ##############################
    dataset = DatasetBuilder.build_kitti_dataset(DatasetBuilder.KITTI_TRAINVAL)

    num_jittered_boxes = 5
    iou_thresh = 0.7

    classes = ['Car']

    sample_name = '000050'

    ##############################

    # Get filtered labels
    dataset.classes = classes
    obj_labels = obj_utils.read_labels(dataset.kitti_label_dir, sample_name)
    obj_labels, class_filter = obj_utils.filter_labels(obj_labels, classes=dataset.classes)

    # Image shape
    bgr_image = cv2.imread(dataset.get_rgb_image_path(sample_name))
    rgb_image = bgr_image[..., :: -1]
    image_shape = rgb_image.shape[0:2]

    # Generate jittered boxes
    aug_labels = []
    for label in obj_labels:
        for i in range(num_jittered_boxes):
            aug_label = kitti_aug.jitter_obj_boxes_2d([label], iou_thresh, image_shape)
            aug_labels.append(aug_label[0])

    # Visualize boxes
    fig, axes = vis_utils.plots_from_image(rgb_image, display=False)

    # Draw non-augmented boxes in red
    for obj in obj_labels:
        vis_utils.draw_obj_as_box_2d(axes, obj, color='r')

    # Draw augmented boxes in cyan
    for obj in aug_labels:
        vis_utils.draw_obj_as_box_2d(axes, obj, color='c', linewidth=1)

    plt.show(block=True)


if __name__ == '__main__':
    main()
