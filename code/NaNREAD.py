import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.io as sio

def get_kdist_neighbors(dist_matrix, k):
    """
    Get k-distance neighbors for all points.
    Based on Definition 2.2.6 & 2.2.7.
    """
    n = dist_matrix.shape[0]
    knns = []
    for i in range(n):
        # Delete self to find k-distance properly
        dists = np.delete(dist_matrix[i], i)
        if len(dists) == 0:
            kdist = 0
        else:
            kdist = np.sort(dists)[k-1]

        # All neighbors <= kdist
        neighbors = set(np.where(dist_matrix[i] <= kdist + 1e-9)[0]) - {i}
        knns.append(neighbors)
    return knns

def natural_neighbor_search(dist_matrix):
    """
    Algorithm 1: Natural Neighbor Search Algorithm.
    Returns lambda (k) and NNAM (Natural Neighbor Adjacency Matrix).
    """
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
                if i in knns[j]: # Mutual kNN
                    NNAM[i, j] = 1
                    NNAM[j, i] = 1

        # Check if all points have at least one natural neighbor
        if np.all(np.sum(NNAM, axis=1) > 0):
            flag = 1
        else:
            k += 1

    return k, NNAM

def compute_NaNE(NNAM):
    """
    Compute Natural Neighbor Entropy (NaNE) given NNAM.
    Definition 2.2.14: NaNE = -(1/n) * sum(log2(|NaNP(oi)|/n))
    """
    n = NNAM.shape[0]
    if n == 0:
        return 0
    sums = np.sum(NNAM, axis=1)

    # Avoid log(0) by using a very small number if sum is 0
    with np.errstate(divide='ignore'):
        log_terms = np.where(sums > 0, np.log2(sums / n), 0)

    return - (1.0 / n) * np.sum(log_terms)

def compute_NaNRE(dist_matrix, base_nane):
    """
    Compute Natural Neighbor Ratio Entropy for all objects.
    Definition 2.2.15.
    NaNRE(oi) = NaNE_oi / NaNE if NaNE_oi / NaNE <= 1 else 1.
    """
    n = dist_matrix.shape[0]
    nanre = np.zeros(n)

    if base_nane == 0:
        return np.ones(n)

    for i in range(n):
        # Remove point i
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
    """
    Compute weight function W_P(oi).
    Definition 2.3.1: W_P(oi) = sqrt(|NaNP(oi)| / |U|)
    """
    n = NNAM.shape[0]
    sums = np.sum(NNAM, axis=1)
    return np.sqrt(sums / n)

def NaNREAD(data):
    """
    Algorithm 2: NaNREAD Algorithm
    """
    n, m = data.shape

    # Pre-compute distance matrices for single attributes and their NaNE
    single_attr_dist = []
    single_attr_nane = []
    single_attr_nnam = []

    for k in range(m):
        attr_data = data[:, [k]]
        dist_mat = squareform(pdist(attr_data, 'euclidean'))
        single_attr_dist.append(dist_mat)

        _, nnam = natural_neighbor_search(dist_mat)
        single_attr_nnam.append(nnam)

        nane = compute_NaNE(nnam)
        single_attr_nane.append(nane)

    # Sort attributes by NaNE (Ascending)
    # Definition 2.3.2: AS = <c'_1, c'_2, ..., c'_m> where NaNE(c'_k) <= NaNE(c'_{k+1})
    sorted_indices = np.argsort(single_attr_nane)

    # Construct AS, AFS, ARS
    # AS: sorted single attributes
    AS = [[idx] for idx in sorted_indices]

    # AFS: Forward sequence C_1, C_2, ... C_m
    AFS = []
    current_afs = []
    for idx in sorted_indices:
        current_afs.append(idx)
        AFS.append(list(current_afs))

    # ARS: Backward sequence C'_1, C'_2, ... C'_m
    # C'_1 = {c'_m}, C'_{k+1} = C'_k U {c'_{m-k}}
    ARS = []
    current_ars = []
    for idx in reversed(sorted_indices):
        current_ars.append(idx)
        ARS.append(list(current_ars))

    # Sequences is a list of lists of attribute indices
    # Calculate NaNRE and Weights for AS, AFS, ARS
    def process_sequence(seq):
        seq_nanre = np.zeros((n, m))
        seq_w = np.zeros((n, m))
        for k, attr_subset in enumerate(seq):
            subset_data = data[:, attr_subset]
            dist_mat = squareform(pdist(subset_data, 'euclidean'))
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

    # Average matrices
    AERM = (ERMAS + ERMAFS + ERMARS) / 3.0
    AWM = (WMAS + WMAFS + WMARS) / 3.0

    # Compute final anomaly score (NaNREAF)
    # NaNREAF(oi) = 1 - sum(AERM[i,k]*AWM[i,k]) / m
    score = np.zeros(n)
    for i in range(n):
        score[i] = 1 - np.sum(AERM[i, :] * AWM[i, :]) / m

    return score

if __name__ == "__main__":
    # Test with Example data
    data_dict = sio.loadmat('code/Example.mat')
    data = data_dict['data']
    print("Data:\n", data)

    # Normalize data to match paper preprocessing in section 2.5 Table 2.1
    # Min-max normalization per column
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)

    # Avoid division by zero if all values in a column are the same
    range_val = data_max - data_min
    range_val[range_val == 0] = 1

    data_norm = (data - data_min) / range_val

    print("Normalized Data:\n", data_norm)

    scores = NaNREAD(data_norm)
    print("NaNREAF Scores:\n", scores)
