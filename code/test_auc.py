import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD

data_dict = sio.loadmat('data/iris_Irisvirginica_11_variant1.mat')
data = data_dict['trandata']
X = data[:, :-1]
y_true = data[:, -1]

scores1 = NaNREAD(X)
auc1 = roc_auc_score(y_true, scores1)
if auc1 < 0.5: auc1 = roc_auc_score(y_true, -scores1)
print("No norm AUC:", auc1)
print("Scores preview:", scores1[:10])
