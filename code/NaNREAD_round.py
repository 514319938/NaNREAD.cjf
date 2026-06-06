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

        # We round the distance matrix directly in the condition!
        # The user hinted: "保留小数点几位的呢"
        neighbors = set(np.where(np.round(dist_matrix[i], 4) <= np.round(kdist, 4) + 1e-9)[0]) - {i}
        knns.append(neighbors)
    return knns

# ... (We will do it inside distance matrix calc directly instead)
