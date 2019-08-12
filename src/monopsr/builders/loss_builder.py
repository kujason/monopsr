import tensorflow as tf

from object_detection.core import losses

from monopsr.core import losses_custom


def get_loss_type_and_weight(loss_config, output_rep):

    if not hasattr(loss_config, output_rep):
        raise ValueError('Loss not configured for output_rep:', output_rep)

    this_loss_config = getattr(loss_config, output_rep)
    loss_type = this_loss_config[0]
    loss_weight = this_loss_config[1]
    return loss_type, loss_weight


def build_loss(loss_type):
    """Builds the desired type of loss

    Args:
        loss_type: loss type (e.g. 'berHu', 'smooth_l1')

    Returns:
        Class of the specified loss_type
    """

    if loss_type == 'berHu':
        return losses_custom.WeightedBerHu()

    elif loss_type == 'chamfer_dist':
        return losses_custom.ChamferDistance()

    elif loss_type == 'emd':
        return losses_custom.EarthMoversDistance()

    elif loss_type == 'smooth_l1':
        return losses.WeightedSmoothL1LocalizationLoss()

    elif loss_type == 'smooth_l1_nonzero':
        return losses_custom.WeightedNonZeroSmoothL1LocalizationLoss()

    elif loss_type == 'softmax':
        return losses.WeightedSoftmaxClassificationLoss()

    elif loss_type == 'focal':
        return losses.SigmoidFocalClassificationLoss()

    elif loss_type == 'softmax_temp':
        return losses.WeightedSoftmaxClassificationLoss(0.5)

    elif loss_type == 'sigmoid_ce':
        return losses_custom.SigmoidClassificationLoss()

    else:
        raise ValueError('Invalid loss type', loss_type)


def add_loss_tensor(loss_config, output_type, pred_tensor, gt_tensor, mask):
    """Add loss tensor multiplied by loss weight

    Args:
        loss_config: Loss config object
        output_type: Output type
        pred_tensor: Prediction tensor
        gt_tensor: Ground truth tensor
        mask: Mask of values to use

    Returns:
        loss_tensor: Loss multiplied by loss weight
    """
    # Get loss configuration
    loss_type, loss_weight = get_loss_type_and_weight(loss_config, output_type)

    if loss_type is None:
        return tf.zeros_like(pred_tensor)

    print('\t{:30s}{}'.format(output_type, loss_type))

    loss_obj = build_loss(loss_type)
    loss_tensor = loss_obj(pred_tensor, gt_tensor, weights=mask)

    return loss_tensor * loss_weight
