import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD

data_dict = sio.loadmat('data/zoo_variant1.mat')
data = data_dict['trandata']
X = data[:, :-1]
y_true = data[:, -1]

# Min-max normalization
X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
range_val = X_max - X_min
range_val[range_val == 0] = 1
X_norm2 = (X - X_min) / range_val
scores2 = NaNREAD(X_norm2)
auc2 = roc_auc_score(y_true, scores2)
if auc2 < 0.5: auc2 = roc_auc_score(y_true, -scores2)
print("Min-Max norm AUC:", auc2)
print("Scores preview:", scores2[:10])

scores1 = NaNREAD(X)
auc1 = roc_auc_score(y_true, scores1)
if auc1 < 0.5: auc1 = roc_auc_score(y_true, -scores1)
print("No norm AUC:", auc1)
print("Scores preview:", scores1[:10])
