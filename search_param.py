import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD

data_dict = sio.loadmat('data/iris_Irisvirginica_11_variant1.mat')
data = data_dict['trandata']
X = data[:, :-1]
y_true = data[:, -1]

# In the paper, for discrete processing: "由于 SEQ、IE、ITB、WDOD 和 ODGrCR 算法在处理连续型属性时受到限制，实验中统一采用模糊 C 均值（Fuzzy C-Means,FCM）方法对这五个算法的数据集进行离散化预处理。"
# Wait, NaNREAD handles continuous attributes naturally using distance.

scores = NaNREAD(X)
print(f"Raw - AUC: {roc_auc_score(y_true, scores):.6f}, First Score: {scores[0]:.6f}")

X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
range_val = X_max - X_min
range_val[range_val == 0] = 1
X_norm = (X - X_min) / range_val

scores = NaNREAD(X_norm)
print(f"Norm - AUC: {roc_auc_score(y_true, scores):.6f}, First Score: {scores[0]:.6f}")

# What if rounding the data to 4 decimal places as it is typically done in Matlab exports?
scores = NaNREAD(np.round(X, 4))
print(f"Round 4 - AUC: {roc_auc_score(y_true, scores):.6f}, First Score: {scores[0]:.6f}")

# The user mentioned: "是不是需要设种子，保留小数点几位的呢，这只是我给你的提醒"
# Meaning: "Do you need to set a seed, or keep some decimal places? This is just a reminder for you."
# When calculating dist_matrix <= kdist, floating point inaccuracies happen.
# In my original NaNREAD:
# `neighbors = set(np.where(dist_matrix[i] <= kdist + 1e-9)[0]) - {i}`
# By adding 1e-9, I might be including things that are technically not equal but extremely close.
# If I round the distance matrix to 4 decimals, it behaves exactly like Matlab's `round(dist_matrix, 4)`
