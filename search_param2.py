import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score

def get_kdist_neighbors(dist_matrix, k):
    n = dist_matrix.shape[0]
    knns = []
    for i in range(n):
        dists = np.delete(dist_matrix[i], i)
        kdist = 0 if len(dists) == 0 else np.sort(dists)[k-1]
        neighbors = set(np.where(np.round(dist_matrix[i], 4) <= round(kdist, 4))[0]) - {i}
        knns.append(neighbors)
    return knns

def natural_neighbor_search(dist_matrix):
    n = dist_matrix.shape[0]
    if n <= 1: return 0, np.zeros((n,n), dtype=int)
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
    if n == 0: return 0
    sums = np.sum(NNAM, axis=1)
    with np.errstate(divide='ignore'):
        log_terms = np.where(sums > 0, np.log2(sums / n), 0)
    return - (1.0 / n) * np.sum(log_terms)

def compute_NaNRE(dist_matrix, base_nane):
    n = dist_matrix.shape[0]
    nanre = np.zeros(n)
    if base_nane == 0: return np.ones(n)
    for i in range(n):
        dist_i = np.delete(np.delete(dist_matrix, i, axis=0), i, axis=1)
        _, nnam_i = natural_neighbor_search(dist_i)
        nane_i = compute_NaNE(nnam_i)
        ratio = nane_i / base_nane
        if 0 <= ratio <= 1: nanre[i] = ratio
        else: nanre[i] = 1.0
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
    from scipy.spatial.distance import pdist, squareform

    for k in range(m):
        attr_data = data[:, [k]]
        dist_mat = squareform(pdist(attr_data, 'euclidean'))
        # Round distance matrix
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

data_dict = sio.loadmat('data/zoo_variant1.mat')
X = data_dict['trandata'][:, :-1]
y_true = data_dict['trandata'][:, -1]

# What happens with NO rounding on distance?
# No Normalization
scores = NaNREAD(X)
auc = roc_auc_score(y_true, scores)
print(f"NO Normalization - AUC: {auc:.6f}, First Score: {scores[0]:.6f}")

# Min-Max Normalization
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
range_val = X_max - X_min
range_val[range_val == 0] = 1
X_norm = (X - X_min) / range_val

scores_norm = NaNREAD(X_norm)
auc_norm = roc_auc_score(y_true, scores_norm)
print(f"Norm - AUC: {auc_norm:.6f}, First Score: {scores_norm[0]:.6f}")
