import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD

data_dict = sio.loadmat('data/iris_Irisvirginica_11_variant1.mat')
X = data_dict['trandata'][:, :-1]
y_true = data_dict['trandata'][:, -1]

# Try Z-score with ddof=0
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0, ddof=0)
X_std[X_std == 0] = 1
X_z = (X - X_mean) / X_std
s = NaNREAD(X_z)
print(f"Z-score ddof=0 -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")

# Try Z-score with ddof=1
X_std1 = np.std(X, axis=0, ddof=1)
X_std1[X_std1 == 0] = 1
X_z1 = (X - X_mean) / X_std1
s = NaNREAD(X_z1)
print(f"Z-score ddof=1 -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")

# Try zscore on the whole matrix instead of per column?
X_mean_all = np.mean(X)
X_std_all = np.std(X)
X_z_all = (X - X_mean_all) / X_std_all
s = NaNREAD(X_z_all)
print(f"Z-score whole matrix -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")

# Try global min max normalization?
X_min_all = np.min(X)
X_max_all = np.max(X)
X_norm_all = (X - X_min_all) / (X_max_all - X_min_all)
s = NaNREAD(X_norm_all)
print(f"MinMax whole matrix -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")
