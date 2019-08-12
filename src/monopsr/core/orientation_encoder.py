import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops


def np_wrap_to_pi(angles):
    """Wrap angles between [-pi, pi]. Angles right at -pi or pi may flip."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def np_orientation_to_angle_bin(orientation, num_bins, overlap):
    """Converts an orientation into an angle bin and residual.
    Example for 8 bins:
         321
        4   0
         567
    Bin centres start at an angle of 0.0.

    Args:
        orientation: orientation angle in radians
        num_bins: number of angle bins
        overlap: amount of overlap for the bins in radians

    Returns:
        angle_bin: bin index
        residual: residual angle from the bin centre
        one_hot_valid_bins: one hot encoding of the valid bins
    """

    two_pi = 2 * np.pi

    # Wrap to [0, 2*pi]
    orientation_wrapped = orientation % two_pi

    angle_per_bin = two_pi / num_bins
    shifted_angle = (orientation_wrapped + angle_per_bin / 2) % two_pi

    best_angle_bin = int(shifted_angle / angle_per_bin)
    best_residual = shifted_angle - (best_angle_bin * angle_per_bin + angle_per_bin / 2)

    # Calculate all residuals for all bin centres
    bin_centres = np.asarray([angle_per_bin * bin_idx for bin_idx in range(num_bins)])
    residuals = np.arctan2(np.sin(orientation_wrapped - bin_centres),
                           np.cos(orientation_wrapped - bin_centres))

    valid_bins = [best_angle_bin]
    if overlap != 0.0:
        # Find which bins indices are valid if they overlap

        # Bin centre of the best bin
        bin_centre = best_angle_bin * angle_per_bin

        # Boundaries of the best bin
        upper_bound = bin_centre + 0.5 * angle_per_bin
        lower_bound = bin_centre - 0.5 * angle_per_bin

        # Calculate distance to the boundaries
        actual_angle = best_angle_bin * angle_per_bin + best_residual
        upper_bound_dist = np.abs(upper_bound - actual_angle)
        lower_bound_dist = np.abs(lower_bound - actual_angle)

        # Determine if the adjacent bins overlap with the actual angle
        if upper_bound_dist < overlap:
            new_valid_bin = best_angle_bin + 1
            if new_valid_bin == num_bins:
                # Wrap to first bin
                new_valid_bin = 0
            valid_bins.append(new_valid_bin)
        elif lower_bound_dist < overlap:
            new_valid_bin = best_angle_bin - 1
            if new_valid_bin < 0:
                # Wrap to last bin
                new_valid_bin = num_bins - 1
                valid_bins.append(new_valid_bin)

    # Create one hot encoding for the valid bins
    one_hot_valid_bins = np.zeros(num_bins)
    one_hot_valid_bins[np.asarray(valid_bins)] = 1

    return best_angle_bin, residuals, one_hot_valid_bins


def np_angle_bin_to_orientation(angle_bin, residual, num_bins):
    """Converts an angle bin and residual into an orientation between [-pi, pi]

    Args:
        angle_bin: bin index
        residual: residual angle from bin centre
        num_bins: number of angle bins

    Returns:
        angle: orientation angle in radians
    """

    two_pi = 2 * np.pi
    angle_per_bin = two_pi / num_bins

    angle_center = angle_bin * angle_per_bin
    angle = angle_center + residual

    # Wrap to [-pi, pi]
    if angle < -np.pi:
        angle = angle + two_pi
    if angle > np.pi:
        angle = angle - two_pi

    return angle


def tf_orientation_to_angle_vector(orientations_tensor):
    """Converts orientation angles into angle unit vector representation.
    e.g. 45 -> [0.717, 0.717], 90 -> [0, 1]

    Args:
        orientations_tensor: A tensor of shape (N,) of orientation angles

    Returns:
        A tensor of shape (N, 2) of angle unit vectors in the format [x, y]
    """
    x = tf.cos(orientations_tensor)
    y = tf.sin(orientations_tensor)

    return tf.stack([x, y], axis=1)


def np_angle_vectors_to_orientations(angle_vectors):
    x = angle_vectors[:, 0]
    y = angle_vectors[:, 1]
    return np.arctan2(y, x)


def tf_angle_vector_to_orientation(angle_vectors_tensor):
    """ Converts angle unit vectors into orientation angle representation.
        e.g. [0.717, 0.717] -> 45, [0, 1] -> 90

    Args:
        angle_vectors_tensor: a tensor of shape (N, 2) of angle unit vectors
            in the format [x, y]

    Returns:
        A tensor of shape (N,) of orientation angles
    """
    x = angle_vectors_tensor[:, 0]
    y = angle_vectors_tensor[:, 1]

    return tf.atan2(y, x)
