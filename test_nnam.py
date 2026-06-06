import numpy as np
import scipy.io as sio
from sklearn.metrics import roc_auc_score
from NaNREAD import NaNREAD

data_dict = sio.loadmat('data/iris_Irisvirginica_11_variant1.mat')
X = data_dict['trandata'][:, :-1]
y_true = data_dict['trandata'][:, -1]

# What if |NaNP(oi)| does NOT include oi?
# In my original compute_NaNE:
# |NaNP(oi)| = row_sum of NNAM.
# And the example verified that this produced exactly the paper's results.
# "Example 2.5: NaNE(c1) = 1.7219"

# Wait, what if we read the dataset "iris_Irisvirginica_11_variant1ori.xls" instead of .mat?
import pandas as pd
df = pd.read_excel('data/iris_Irisvirginica_11_variant1ori.xls')
print("Ori XLS shape:", df.shape)
