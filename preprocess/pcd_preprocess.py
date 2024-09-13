import numpy as np
from hdbscan import HDBSCAN
from collections import defaultdict


def overlap_clusters(cluster_i, cluster_j, min_cluster_point=10):
    # get unique labels from pcd_i and pcd_j from segments bigger than min_clsuter_point
    unique_i, count_i = np.unique(cluster_i, return_counts=True)
    unique_i = unique_i[count_i > min_cluster_point]

    unique_j, count_j = np.unique(cluster_j, return_counts=True)
    unique_j = unique_j[count_j > min_cluster_point]

    # get labels present on both pcd (intersection)
    unique_ij = np.intersect1d(unique_i, unique_j)[1:]

    # labels not intersecting both pcd are assigned as -1 (unlabeled)
    cluster_i[np.in1d(cluster_i, unique_ij, invert=True)] = -1
    cluster_j[np.in1d(cluster_j, unique_ij, invert=True)] = -1

    return cluster_i, cluster_j


def apply_transform(points, pose):
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    return np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)[:, :3]


def undo_transform(points, pose):
    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
    return np.sum(np.expand_dims(hpoints, 2) * np.linalg.inv(pose).T, axis=1)[:, :3]


def grid_sample(points_set, resolution, temporal_weight=1):

    points_sparse = np.round(points_set[:, :4] / resolution)
    points_sparse -= points_sparse.min(0, keepdims=1)
    points_sparse = points_sparse.astype(np.int32)
    points_sparse, indexes, inverse_indexes = np.unique(points_sparse[:, :4], axis=0, return_index=True, return_inverse=True)
    points = np.zeros((len(points_sparse), 4), dtype=np.float32)
    np.add.at(points, inverse_indexes, points_set)
    _, count = np.unique(inverse_indexes, return_counts=True)
    points = points / count[:, None]
    points[:, 3] *= temporal_weight
    return points, indexes, inverse_indexes

    points_set[:, :4] = np.round(points_set[:, :4] / resolution)
    points, indexes, inverse_indexes = np.unique(points_set[:, :4], axis=0, return_index=True, return_inverse=True)
    points[:, :4] -= points[:, :4].min(0, keepdims=1)
    points = np.hstack([points, points_set[indexes, 4:]])
    points[:, 3] *= temporal_weight
    return points, indexes, inverse_indexes


def clusters_hdbscan(points_set, **kwargs):
    if "min_cluster_size" not in kwargs:
        kwargs["min_cluster_size"] = 20
    clusterer = HDBSCAN(**kwargs, gen_min_span_tree=False, leaf_size=100, core_dist_n_jobs=1)
    clusterer.fit(points_set)

    labels = clusterer.labels_
    labels[labels>=0] +=1  # -1 is for unlabeled

    return labels


def clusterize_pcd(points, ground, **kwargs):
    pcd_ = points[~ground, :4]
    labels_ = clusters_hdbscan(pcd_, **kwargs)

    labels = np.full(points.shape[0], -1, dtype=np.int16)

    labels[ground] = 0
    labels[~ground] = labels_

    return labels


def order_segments(labels, min_points=None):
    # we have to deal separately with the ground and -1
    lbls, counts = np.unique(labels, return_counts=True)
    cluster_info = np.array(list(zip(lbls[2:], counts[2:])))  # -1 and 0 are not clusters
    cluster_info = cluster_info[cluster_info[:, 1].argsort()]

    clusters_labels = cluster_info[::-1][:, 0]
    if min_points is not None:
        num_labels = len(cluster_info) - np.argmax(cluster_info[:, 1] >= min_points)
        mapping = defaultdict(lambda: -1, zip(clusters_labels[:num_labels], np.arange(1, num_labels + 1)))
    else:
        num_labels = len(cluster_info)
        mapping = dict(zip(clusters_labels[:num_labels], np.arange(1, num_labels + 1)))
        mapping[-1] = -1
    mapping[0] = 0

    labels = np.vectorize(mapping.__getitem__)(labels).astype(np.int16)
    return labels
