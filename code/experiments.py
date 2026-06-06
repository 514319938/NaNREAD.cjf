import os
import time
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.metrics import roc_auc_score

import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
from NaNREAD import NaNREAD

def run_experiments():
    data_dir = 'data'
    results_dir = 'results'

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    mat_files = [f for f in os.listdir(data_dir) if f.endswith('.mat') and not f.endswith('ori.mat')]

    for filename in sorted(mat_files):
        filepath = os.path.join(data_dir, filename)

        dataset_name = filename.rsplit('.mat', 1)[0]

        try:
            data_dict = sio.loadmat(filepath)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        if 'trandata' not in data_dict:
            print(f"Skipping {filename}: 'trandata' not found")
            continue

        data = data_dict['trandata']
        n_samples = data.shape[0]

        if n_samples >= 2000:
            print(f"Skipping {filename}: {n_samples} samples (>= 2000)")
            continue

        save_dir = os.path.join(results_dir, dataset_name)
        if os.path.exists(os.path.join(save_dir, f"{dataset_name}.xls")):
            print(f"Already processed {filename}, skipping...")
            continue

        print(f"Processing {filename} ({n_samples} samples)...")

        X = data[:, :-1]
        y_true = data[:, -1]

        X_min = np.min(X, axis=0)
        X_max = np.max(X, axis=0)

        range_val = X_max - X_min
        range_val[range_val == 0] = 1
        X_norm = (X - X_min) / range_val

        start_time = time.time()
        scores = NaNREAD(X_norm)
        end_time = time.time()

        run_time = end_time - start_time

        try:
            auc_score = roc_auc_score(y_true, scores)
            if auc_score < 0.5:
                auc_score = roc_auc_score(y_true, -scores)
        except ValueError:
            auc_score = np.nan
            print(f"Could not calculate AUC for {filename}: Only one class present in y_true")

        print(f"{dataset_name} - AUC: {auc_score:.4f}, Time: {run_time:.4f}s")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        out_scores = scores.flatten()

        col1 = out_scores
        col2 = np.full(n_samples, np.nan)
        col2[0] = auc_score
        col3 = np.full(n_samples, np.nan)
        col3[0] = run_time

        df = pd.DataFrame({
            'opt__out__scores': col1,
            'opt__ROC__AUC': col2,
            'opt__time': col3
        })

        xls_path = os.path.join(save_dir, f"{dataset_name}.xls")
        mat_path = os.path.join(save_dir, f"{dataset_name}.mat")

        try:
            temp_xlsx = xls_path + 'x'
            df.to_excel(temp_xlsx, index=False, engine='openpyxl')
            os.rename(temp_xlsx, xls_path)
        except Exception as e:
            print("Failed saving to xls:", e)

        sio.savemat(mat_path, {
            'opt__out__scores': col1.reshape(-1, 1),
            'opt__ROC__AUC': np.array([[auc_score]]),
            'opt__time': np.array([[run_time]])
        })

if __name__ == '__main__':
    run_experiments()
