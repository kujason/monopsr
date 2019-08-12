import tensorflow as tf


def tf_normalize_cen_y_by_mean(unnormalized_cen_y, class_strs):
    """Normalizes the centroid y position by dividing by the mean centroid y value of the class.

    Args:
        unnormalized_cen_y: Unnormalized y centroid
        class_strs: tf.string of the class

    Returns:
        normalized_cen_y: Normalized y centroid
    """

    # See box_means.py
    def cen_y_mean_car():
        return 1.7153475

    def cen_y_mean_pedestrian():
        return 1.4557862

    def cen_y_mean_cyclist():
        return 1.5591882

    class_strs = tf.squeeze(class_strs)

    mean_cen_y = tf.map_fn(
        lambda x: tf.case({
            tf.equal(x, 'Car'): cen_y_mean_car,
            tf.equal(x, 'Pedestrian'): cen_y_mean_pedestrian,
            tf.equal(x, 'Cyclist'): cen_y_mean_cyclist}), class_strs,
        dtype=tf.float32)

    normalized_cen_y = unnormalized_cen_y / tf.expand_dims(mean_cen_y, 1)

    return normalized_cen_y


def tf_normalize_cen_z_by_mean(unnormalized_cen_z, class_strs):
    """Normalizes the centroid z position by dividing by the mean centroid z value of the class.

    Args:
        unnormalized_cen_z: Unnormalized z centroid
        class_strs: tf.string of the class

    Returns:
        normalized_cen_z: Normalized z centroid
    """

    # See box_means.py
    def cen_z_mean_car():
        return 25.24178

    def cen_z_mean_pedestrian():
        return 17.95974

    def cen_z_mean_cyclist():
        return 21.279533

    class_strs = tf.squeeze(class_strs)

    mean_cen_z = tf.map_fn(
        lambda x: tf.case({
            tf.equal(x, 'Car'): cen_z_mean_car,
            tf.equal(x, 'Pedestrian'): cen_z_mean_pedestrian,
            tf.equal(x, 'Cyclist'): cen_z_mean_cyclist}), class_strs,
        dtype=tf.float32)

    normalized_cen_z = unnormalized_cen_z / tf.expand_dims(mean_cen_z, 1)

    return normalized_cen_z


def tf_normalize_box_height_by_mean(unnormalized_box_height, class_strs):
    """Normalizes the 2D box height by dividing by the mean box height value of the class.

    Args:
        unnormalized_box_height: Unnormalized box height
        class_strs: tf.string of the class

    Returns:
        normalized_box_height: Normalized box height
    """

    # See box_means.py
    def box_h_mean_car():
        return 61.594734

    def box_h_mean_pedestrian():
        return 95.95055

    def box_h_mean_cyclist():
        return 76.85717

    class_strs = tf.squeeze(class_strs)

    mean_box_h = tf.map_fn(
        lambda x: tf.case({
            tf.equal(x, 'Car'): box_h_mean_car,
            tf.equal(x, 'Pedestrian'): box_h_mean_pedestrian,
            tf.equal(x, 'Cyclist'): box_h_mean_cyclist}), class_strs,
        dtype=tf.float32)

    normalized_box_height = unnormalized_box_height / tf.expand_dims(mean_box_h, 1)

    return normalized_box_height
