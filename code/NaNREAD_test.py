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

        # What if we round distance matrix to 4 decimals BEFORE comparison?
        # The user hinted: "保留小数点几位的呢" -> "keep a few decimal places"
        # Let's round the distance matrix to 4 decimals and check equality:
        # Matlab's pdist euclidean returns double precision, but if they rounded it...

        neighbors = set(np.where(np.round(dist_matrix[i], 4) <= np.round(kdist, 4))[0]) - {i}
        knns.append(neighbors)
    return knns

# ... (We will just modify the distance matrix directly instead of in get_kdist_neighbors)
