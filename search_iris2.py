import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD
from NaNREAD_dist4 import NaNREAD as NaNREAD_dist4

data_dict = sio.loadmat('data/iris_Irisvirginica_11_variant1.mat')
X = data_dict['trandata'][:, :-1]
y_true = data_dict['trandata'][:, -1]

# What if we round the distance matrix to exactly 4 decimals (using NaNREAD_dist4)?
s = NaNREAD_dist4(X)
print(f"NoNorm Dist4 -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")

# What if we round data to 4 decimals?
s = NaNREAD(np.round(X, 4))
print(f"Data Round4 -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")

# What if we use L1 distance instead of L2? (euclidean)
from scipy.spatial.distance import pdist, squareform
def NaNREAD_cityblock(data):
    n, m = data.shape
    single_attr_dist = []
    single_attr_nane = []
    single_attr_nnam = []
    from NaNREAD import natural_neighbor_search, compute_NaNE, compute_NaNRE, compute_weight
    for k in range(m):
        attr_data = data[:, [k]]
        dist_mat = squareform(pdist(attr_data, 'cityblock'))
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
            dist_mat = squareform(pdist(subset_data, 'cityblock'))
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

s = NaNREAD_cityblock(X)
print(f"L1 distance -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")
