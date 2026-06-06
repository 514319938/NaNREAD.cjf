import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD

# The paper explicitly states: "由于 SEQ、IE、ITB、WDOD 和 ODGrCR 算法在处理连续型属性时受到限制，实验中统一采用模糊 C 均值（Fuzzy C-Means,FCM）方法对这五个算法的数据集进行离散化预处理。"
# It does NOT say FCM was used for NaNREAD. But what if it WAS? Or maybe something else?

# Wait! Look closely at the image provided by the user.
# image 3: opt__out_scores for iris_Irisvirginica_11_variant1
# First score is 0.656372, AUC is 0.937273
# Can we get exactly these numbers by rounding to some specific decimal? Or doing a different normalization?

data_dict = sio.loadmat('data/iris_Irisvirginica_11_variant1.mat')
X = data_dict['trandata'][:, :-1]
y_true = data_dict['trandata'][:, -1]

# Try scaling the data to [0,1] based on column max?
X_max = np.max(X, axis=0)
X_max[X_max == 0] = 1
X_norm_max = X / X_max

s = NaNREAD(X_norm_max)
print(f"Norm Max -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")

# Try standardizing using L2 norm
X_norm_l2 = X / np.linalg.norm(X, axis=0)
s = NaNREAD(X_norm_l2)
print(f"Norm L2 -> Score1: {s[0]:.6f}, AUC: {roc_auc_score(y_true, s):.6f}")

# Wait, check if FCM discretization could actually be used for NaNREAD in the authors' codebase despite what the paper says.
# But FCM would convert continuous data into discrete cluster memberships.
