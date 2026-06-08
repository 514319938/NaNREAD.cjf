import os
import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, f1_score

def load_ground_truth(dataset_name, data_dir='data'):
    mat_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if not os.path.exists(mat_path):
        return None
    try:
        data_dict = sio.loadmat(mat_path)
        if 'trandata' in data_dict:
            data = data_dict['trandata']
            y_true = data[:, -1]
            return y_true
        else:
            return None
    except Exception as e:
        return None

def load_scores(dataset_name, alg, results_base_dir, nanread_base_dir):
    if alg == 'NaNREAD':
        mat_path = os.path.join(nanread_base_dir, dataset_name, f"{dataset_name}.mat")
    else:
        alg_dir = f"{alg}_results"
        mat_path = os.path.join(results_base_dir, alg_dir, dataset_name, f"{dataset_name}_{alg}.mat")

    if not os.path.exists(mat_path):
        return None

    try:
        mat_data = sio.loadmat(mat_path)
        if 'opt_out_scores' in mat_data:
            return mat_data['opt_out_scores'].flatten()
        elif 'opt__out__scores' in mat_data:
            return mat_data['opt__out__scores'].flatten()
        else:
            return None
    except Exception as e:
        return None

def main():
    datasets = [
        'breast_cancer_variant1', 'chess_nowin_34_variant1', 'monks_0_4_variant1',
        'tic_tac_toe_negative_12_variant1', 'tic_tac_toe_negative_26_variant1',
        'zoo_variant1', 'cardiotocography_2and3_33_variant1',
        'diabetes_tested_positive_26_variant1', 'glass', 'ionosphere_b_24_variant1',
        'letter', 'pima_TRUE_55_variant1', 'vowels', 'annealing_variant1',
        'bands_band_6_variant1'
    ]

    algorithms = [
        'SEQ', 'IE', 'ITB', 'WDOD', 'ODGrCR', 'ApproE', 'VarE',
        'ILGNI', 'MFIOD', 'VAE', 'NaNREAD'
    ]

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    nanread_base_dir = os.path.join(base_dir, 'results')

    if os.name == 'nt':
        results_base_dir = r'F:\Experiental_results'
    else:
        results_base_dir = '/tmp/Experiental_results'

    f1_results = []

    for ds in datasets:
        y_true = load_ground_truth(ds, data_dir)
        if y_true is None:
            print(f"Skipping {ds}, ground truth not found.")
            continue

        # Determine anomaly class as the minority class
        unique_labels, counts = np.unique(y_true, return_counts=True)
        outlier_label = unique_labels[np.argmin(counts)]
        k = np.min(counts)

        y_binary = (y_true == outlier_label).astype(int)

        row_data = {'Dataset': ds}

        for alg in algorithms:
            scores = load_scores(ds, alg, results_base_dir, nanread_base_dir)
            if scores is not None:
                # Determine direction: if AUC < 0.5, invert scores
                fpr, tpr, _ = roc_curve(y_binary, scores)
                roc_auc = auc(fpr, tpr)
                if roc_auc < 0.5:
                    scores = -scores

                # Top k as outliers
                indices = np.argsort(scores)[-k:]
                y_pred = np.zeros_like(y_binary)
                y_pred[indices] = 1

                f1 = f1_score(y_binary, y_pred)
                row_data[alg] = f1
            else:
                row_data[alg] = np.nan

        f1_results.append(row_data)

    df_f1 = pd.DataFrame(f1_results)

    output_dir = os.path.join(base_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'F1_scores.xls')

    try:
        temp_xlsx = output_file + 'x'
        df_f1.to_excel(temp_xlsx, index=False, engine='openpyxl')
        if os.path.exists(output_file):
            os.remove(output_file)
        os.rename(temp_xlsx, output_file)
        print(f"Successfully saved F1 scores to {output_file}")
    except Exception as e:
        print(f"Failed to save {output_file}: {e}")

if __name__ == '__main__':
    main()
