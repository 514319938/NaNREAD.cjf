import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.io as sio

def get_kdist_neighbors(dist_matrix, k):
    n = dist_matrix.shape[0]
    knns = []
    for i in range(n):
        dists = np.delete(dist_matrix[i], i)
        if len(dists) == 0:
            kdist = 0
        else:
            kdist = np.sort(dists)[k-1]

        # User specified to round distance to avoid precision mismatch, typically to 4 decimals
        neighbors = set(np.where(np.round(dist_matrix[i], 4) <= np.round(kdist, 4) + 1e-9)[0]) - {i}
        knns.append(neighbors)
    return knns

def natural_neighbor_search(dist_matrix):
    n = dist_matrix.shape[0]
    if n <= 1:
        return 0, np.zeros((n,n), dtype=int)

    NNAM = np.zeros((n, n), dtype=int)
    k = 1
    flag = 0

    while flag == 0 and k <= n - 1:
        knns = get_kdist_neighbors(dist_matrix, k)

        for i in range(n):
            for j in knns[i]:
                if i in knns[j]:
                    NNAM[i, j] = 1
                    NNAM[j, i] = 1

        if np.all(np.sum(NNAM, axis=1) > 0):
            flag = 1
        else:
            k += 1

    return k, NNAM

def compute_NaNE(NNAM):
    n = NNAM.shape[0]
    if n == 0:
        return 0
    sums = np.sum(NNAM, axis=1)

    with np.errstate(divide='ignore'):
        log_terms = np.where(sums > 0, np.log2(sums / n), 0)

    return - (1.0 / n) * np.sum(log_terms)

def compute_NaNRE(dist_matrix, base_nane):
    n = dist_matrix.shape[0]
    nanre = np.zeros(n)

    if base_nane == 0:
        return np.ones(n)

    for i in range(n):
        dist_i = np.delete(np.delete(dist_matrix, i, axis=0), i, axis=1)
        _, nnam_i = natural_neighbor_search(dist_i)
        nane_i = compute_NaNE(nnam_i)

        ratio = nane_i / base_nane
        if 0 <= ratio <= 1:
            nanre[i] = ratio
        else:
            nanre[i] = 1.0

    return nanre

def compute_weight(NNAM):
    n = NNAM.shape[0]
    sums = np.sum(NNAM, axis=1)
    return np.sqrt(sums / n)

def NaNREAD(data):
    n, m = data.shape

    single_attr_dist = []
    single_attr_nane = []
    single_attr_nnam = []

    for k in range(m):
        attr_data = data[:, [k]]
        dist_mat = squareform(pdist(attr_data, 'euclidean'))

        # We round the matrix itself to 4 decimals here
        dist_mat = np.round(dist_mat, 4)

        single_attr_dist.append(dist_mat)

        _, nnam = natural_neighbor_search(dist_mat)
        single_attr_nnam.append(nnam)

        nane = compute_NaNE(nnam)
        single_attr_nane.append(nane)

    sorted_indices = np.argsort(single_attr_nane)

    AS = [[idx] for idx in sorted_indices]

    AFS = []
    current_afs = []
    for idx in sorted_indices:
        current_afs.append(idx)
        AFS.append(list(current_afs))

    ARS = []
    current_ars = []
    for idx in reversed(sorted_indices):
        current_ars.append(idx)
        ARS.append(list(current_ars))

    def process_sequence(seq):
        seq_nanre = np.zeros((n, m))
        seq_w = np.zeros((n, m))
        for k, attr_subset in enumerate(seq):
            subset_data = data[:, attr_subset]
            dist_mat = squareform(pdist(subset_data, 'euclidean'))

            # Round matrix to 4 decimals
            dist_mat = np.round(dist_mat, 4)

            _, nnam = natural_neighbor_search(dist_mat)
            nane = compute_NaNE(nnam)

            nanre = compute_NaNRE(dist_mat, nane)
            w = compute_weight(nnam)

            seq_nanre[:, k] = nanre
            seq_w[:, k] = w
        return seq_nanre, seq_w

    ERMAS, WMAS = process_sequence(AS)
    ERMAFS, WMAFS = process_sequence(AFS)
    ERMARS, WMARS = process_sequence(ARS)

    AERM = (ERMAS + ERMAFS + ERMARS) / 3.0
    AWM = (WMAS + WMAFS + WMARS) / 3.0

    score = np.zeros(n)
    for i in range(n):
        score[i] = 1 - np.sum(AERM[i, :] * AWM[i, :]) / m

    return score
