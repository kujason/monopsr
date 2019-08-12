import tensorflow as tf


class ImgPreprocessor:

    # Kitti image mean per channel
    _KITTI_CHANNEL_MEANS = [92.8403, 97.7996, 93.5843]

    # ImageNet image mean per channel
    _IMAGENET_CHANNEL_MEANS = [123.68, 116.78, 103.94]

    def preprocess_input(self, tensor_in, output_size, mean_sub_type):
        """Preprocesses by performing mean subtraction and resizing.

        Args:
            tensor_in: A `Tensor` of shape=(batch_size, height,
                width, channels) representing an input image.
            output_size: The size of the input (H x W)
            mean_sub_type: Type of mean subtraction to apply (KITTI or imagenet)

        Returns:
            Tensor input mean subtracted and resized to the output_size
        """
        image = tf.to_float(tensor_in)
        if mean_sub_type == 'kitti':
            channel_means = self._KITTI_CHANNEL_MEANS
        elif mean_sub_type == 'imagenet':
            channel_means = self._IMAGENET_CHANNEL_MEANS
        else:
            raise ValueError('Invalid mean subtraction type {}'.format(mean_sub_type))
        image_centered = image - channel_means
        image_resized = tf.image.resize_images(image_centered, output_size)

        tensor_out = image_resized
        return tensor_out
