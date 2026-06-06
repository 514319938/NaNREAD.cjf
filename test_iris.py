import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD

data_dict = sio.loadmat('data/iris_Irisvirginica_11_variant1.mat')
X = data_dict['trandata'][:, :-1]
y_true = data_dict['trandata'][:, -1]

# L1 normalization?
# In Table 2.3 of the paper, the AUC for Iono is 0.9998, Monk is 1.000, Zoo is 0.6520
# Let's check Zoo AUC without normalization and with round
data_zoo = sio.loadmat('data/zoo_variant1.mat')['trandata']
X_z = data_zoo[:, :-1]
y_z = data_zoo[:, -1]

# Min Max -> Round 4
def do_eval(X, y, name):
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    r = X_max - X_min
    r[r==0] = 1
    X_n = np.round((X - X_min)/r, 4)
    s = NaNREAD(X_n)
    auc = roc_auc_score(y, s)
    if auc < 0.5: auc = roc_auc_score(y, -s)
    print(f"{name} AUC: {auc:.6f}")
    print(f"{name} s: {s[:5]}")

do_eval(X_z, y_z, "Zoo")
