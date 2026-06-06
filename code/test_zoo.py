import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD

data_dict = sio.loadmat('data/zoo_variant1.mat')
data = data_dict['trandata']
X = data[:, :-1]
y_true = data[:, -1]

# We need to test if rounding the NORMALIZED DATA itself to 4 decimals gets the exact answer.
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
range_val = X_max - X_min
range_val[range_val == 0] = 1
X_norm = (X - X_min) / range_val
X_norm_round4 = np.round(X_norm, 4)

scores = NaNREAD(X_norm_round4)
auc = roc_auc_score(y_true, scores)
if auc < 0.5: auc = roc_auc_score(y_true, -scores)
print("Norm Round 4 AUC:", auc)
print("Scores preview:", scores[:10])

# What if rounding the raw data to 4 decimals?
X_round4 = np.round(X, 4)
scores2 = NaNREAD(X_round4)
auc2 = roc_auc_score(y_true, scores2)
if auc2 < 0.5: auc2 = roc_auc_score(y_true, -scores2)
print("No Norm Round 4 AUC:", auc2)
print("Scores preview:", scores2[:10])
