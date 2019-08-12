import tensorflow as tf

from object_detection.core.losses import Loss
from tf_ops.approxmatch import tf_approxmatch
from tf_ops.nn_distance import tf_nndistance


class BerHu(Loss):
    """berHu Loss
    """

    def _compute_loss(self, prediction_tensor, target_tensor):
        """Compute loss function.

        berHu loss:
        l1_thresh = 1/5 * maximum pixel wise difference in a batch

        L1 distance when the error is less than or equal to l1_thresh,
        (error^2 + l1_thresh^2) / 2*l1_thresh when error is greater than
        l1_thresh

        Args:
            prediction_tensor: A float tensor of shape [batch_size, h, w, c]
            target_tensor: A float tensor of shape [batch_size, h, w, c]
        Returns:
          loss_normalized: berHu error between depth gt image and predicted depth
        """

        error = tf.subtract(prediction_tensor, target_tensor)
        abs_error = tf.abs(error)

        l1_thresh = tf.reduce_max(abs_error) / 5.0

        # Case 1: |x| <= l1_thresh
        case1_error = abs_error
        # Case 2: (x^2 + l1_thresh^2) / (2*l1_thresh)
        case2_error = (error**2 + l1_thresh**2) / (2 * l1_thresh)

        condition = tf.less_equal(abs_error, l1_thresh)
        loss = tf.where(condition, case1_error, case2_error)

        return loss


class WeightedBerHu(Loss):
    """Weighted berHu loss
    """

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        berHu loss:
        l1_thresh = 1/5 * maximum pixel wise difference in a batch

        L1 distance when the error is less than or equal to l1_thresh,
        (error^2 + l1_thresh^2) / 2*l1_thresh when error is greater than
        l1_thresh

        Args:
            prediction_tensor: A float tensor of shape [batch_size, h, w, c]
            target_tensor: A float tensor of shape [batch_size, h, w, c]
            weights (float): A tensor of shape [batch_size, h, w, 1]
        Returns:
          loss_normalized: berHu error between depth gt image and predicted depth
        """

        error = tf.subtract(prediction_tensor, target_tensor)
        abs_error = tf.abs(error)

        l1_thresh = tf.reduce_max(abs_error) / 5.0

        # Case 1: |x| <= l1_thresh
        case1_error = abs_error
        # Case 2: (x^2 + l1_thresh^2) / (2*l1_thresh)
        case2_error = (error**2 + l1_thresh**2) / (2 * l1_thresh)

        condition = tf.less_equal(abs_error, l1_thresh)
        loss_per_pixel = tf.where(condition, case1_error, case2_error)

        # Ignore pixels <= 0 and sum loss
        loss = tf.reduce_sum(loss_per_pixel * tf.to_float(weights))

        # Check number of valid pixels and normalize loss
        num_valid = tf.to_float(tf.count_nonzero(weights))
        valid_cond = tf.greater(num_valid, 0)
        loss_normalized = tf.cond(valid_cond,
                                  lambda: loss / num_valid,
                                  lambda: tf.constant(0.0))

        return loss_normalized


class WeightedNonZeroSmoothL1LocalizationLoss(Loss):
    """Smooth L1 localization loss function aka Huber Loss..

    The smooth L1_loss is defined elementwise as .5 x^2 if |x| <= delta and
    0.5 x^2 + delta * (|x|-delta) otherwise, where x is the difference between
    predictions and target.

    See also Equation (3) in the Fast R-CNN paper by Ross Girshick (ICCV 2015)
    """

    def __init__(self, delta=1.0):
        """Constructor.

        Args:
          delta: delta for smooth L1 loss.
        """
        self._delta = delta

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """Compute loss function.

        Args:
          prediction_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the (encoded) predicted locations of objects.
          target_tensor: A float tensor of shape [batch_size, num_anchors,
            code_size] representing the regression targets
          weights: a float tensor of shape [batch_size, num_anchors]

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors] tensor
            representing the value of the loss function.
        """
        return tf.losses.huber_loss(
            target_tensor,
            prediction_tensor,
            delta=self._delta,
            weights=weights,
            loss_collection=None,
            reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        )


class EarthMoversDistance(Loss):
    """Approximation of the Earth Mover's Distance that compares two point clouds
    """

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """

        Args:
            prediction_tensor (float): A tensor of shape [batch_size, h, w, 3]
            target_tensor (float): A tensor of shape [batch_size, h, w, 3]
            weights (float): A tensor of shape [batch_size, h, w, 1]

        Returns:
            loss (float): Approximate Earth Mover's Distance
        """

        # Multiply by valid mask
        valid_prediction_tensor = prediction_tensor * weights
        valid_target_tensor = target_tensor * weights

        # Reshape to (batch_size, n_points, 3)
        batch_size = prediction_tensor.get_shape()[0]
        valid_prediction_points = tf.reshape(valid_prediction_tensor, (batch_size, -1, 3))
        valid_target_points = tf.reshape(valid_target_tensor, (batch_size, -1, 3))

        match = tf_approxmatch.approx_match(valid_prediction_points, valid_target_points)
        distances = tf_approxmatch.match_cost(
            valid_prediction_points, valid_target_points, match)
        emd_distance = tf.reduce_sum(distances) / tf.cast(batch_size, tf.float32)

        return emd_distance


class ChamferDistance(Loss):
    """Computes the chamfer distance between two point clouds
    """

    def _compute_loss(self, prediction_tensor, target_tensor, weights):
        """

        Args:
            prediction_tensor (float): A tensor of shape [batch_size, h, w, 3]
            target_tensor (float): A tensor of shape [batch_size, h, w, 3]
            weights (float): A tensor of shape [batch_size, h, w, 1]

        Returns:
            loss (float): chamfer distance
        """

        # Multiply by valid mask
        valid_prediction_tensor = prediction_tensor * weights
        valid_target_tensor = target_tensor * weights

        # Reshape to (batch_size, n_points, 3)
        batch_size = prediction_tensor.get_shape()[0]
        valid_prediction_points = tf.reshape(valid_prediction_tensor, (batch_size, -1, 3))
        valid_target_points = tf.reshape(valid_target_tensor, (batch_size, -1, 3))

        dist1, idx1, dist2, idx2 = tf_nndistance.nn_distance(
            valid_prediction_points, valid_target_points)
        chamfer_dist = tf.reduce_sum(dist1) + tf.reduce_sum(dist2)
        avg_chamfer_dist = chamfer_dist / tf.cast(batch_size, tf.float32)

        return avg_chamfer_dist


class SigmoidClassificationLoss(Loss):
    """Sigmoid cross entropy classification loss function."""

    def _compute_loss(self,
                      prediction_tensor,
                      target_tensor,
                      class_indices=None,
                      weights=None):
        """Compute loss function.

        Args:
            prediction_tensor: A float tensor of shape [batch_size, ..., num_classes]
                representing the predicted logits for each class
            target_tensor: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
            class_indices: (Optional) A 1-D integer tensor of class indices.
            If provided, computes loss only for the specified class indices.
            weights: Not used

        Returns:
          loss: a float tensor of shape [batch_size, num_anchors, num_classes]
            representing the value of the loss function.
        """
        # if class_indices is not None:
        #     weights *= tf.reshape(
        #         ops.indices_to_dense_vector(class_indices,
        #                                     tf.shape(prediction_tensor)[2]),
        #         [1, 1, -1])
        per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
            labels=target_tensor, logits=prediction_tensor))
        return per_entry_cross_ent
        # return per_entry_cross_ent * weights
