import numpy as np
from sklearn.neighbors import NearestNeighbors


def calc_chamfer_dist(points_1, points_2):
    """Calculates chamfer distance between two sets of points

    Args:
        points_1: (N, 3) points
        points_2: (N, 3) points

    Returns:
        chamfer_distance: chamfer distance between two sets of points
    """
    nns_1 = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points_1)
    nns_2 = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(points_2)

    nn_1_dists, nn_1_indices = nns_1.kneighbors(points_2)
    nn_2_dists, nn_2_indices = nns_2.kneighbors(points_1)

    chamfer_distance = np.sum(nn_1_dists**2) + np.sum(nn_2_dists**2)

    return chamfer_distance
