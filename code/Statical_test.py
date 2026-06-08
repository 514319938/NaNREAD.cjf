import os
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.stats import rankdata
import math

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
    except:
        return None
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
    except:
        return None
    return None

def plot_cd_diagram(avranks, num_datasets, cd, output_path):
    """
    Plots the Critical Distance diagram based on average ranks.
    """
    algorithms = list(avranks.keys())
    ranks = list(avranks.values())
    n_algs = len(algorithms)

    # Sort algorithms by their rank
    sorted_indices = np.argsort(ranks)
    algorithms = [algorithms[i] for i in sorted_indices]
    ranks = [ranks[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(10, max(4, n_algs * 0.5)))

    ax.set_xlim(1, n_algs)
    ax.set_ylim(0, n_algs + 1)

    # Hide all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Axis line at the top
    ax.plot([1, n_algs], [n_algs, n_algs], color='k', linewidth=1)

    # Ticks
    for i in range(1, n_algs + 1):
        ax.plot([i, i], [n_algs, n_algs + 0.1], color='k', linewidth=1)
        ax.text(i, n_algs + 0.2, str(i), ha='center', va='bottom')

    # CD line
    ax.plot([1, 1 + cd], [n_algs + 1, n_algs + 1], color='r', linewidth=2)
    ax.text(1 + cd/2, n_algs + 1.2, f'CD = {cd:.4f}', ha='center', color='r')

    left_algs = algorithms[:math.ceil(n_algs/2)]
    right_algs = algorithms[math.ceil(n_algs/2):]

    left_ranks = ranks[:math.ceil(n_algs/2)]
    right_ranks = ranks[math.ceil(n_algs/2):]

    y_pos = n_algs - 1

    for alg, rank in zip(left_algs, left_ranks):
        ax.plot([rank, rank], [n_algs, y_pos], color='k', linewidth=1)
        ax.plot([rank, 1], [y_pos, y_pos], color='k', linewidth=1)
        ax.text(0.9, y_pos, f'{alg} ({rank:.2f})', ha='right', va='center')
        y_pos -= 1

    y_pos = n_algs - 1
    for alg, rank in zip(right_algs, right_ranks):
        ax.plot([rank, rank], [n_algs, y_pos], color='k', linewidth=1)
        ax.plot([rank, n_algs], [y_pos, y_pos], color='k', linewidth=1)
        ax.text(n_algs + 0.1, y_pos, f'{alg} ({rank:.2f})', ha='left', va='center')
        y_pos -= 1

    # Draw connections between algorithms that are not significantly different
    # (Simplified visualization for Nemenyi CD)
    for i in range(n_algs):
        for j in range(i+1, n_algs):
            if ranks[j] - ranks[i] <= cd:
                ax.plot([ranks[i], ranks[j]], [n_algs - 0.5 - i*0.2, n_algs - 0.5 - i*0.2],
                        color='r', linewidth=2, alpha=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Critical Distance Diagram (Nemenyi Test)", y=1.2)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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
    output_dir = os.path.join(base_dir, 'results_figures')

    if os.name == 'nt':
        results_base_dir = r'F:\Experiental_results'
    else:
        results_base_dir = '/tmp/Experiental_results'

    auc_results = {}

    for ds in datasets:
        y_true = load_ground_truth(ds, data_dir)
        if y_true is None:
            continue

        unique_labels, counts = np.unique(y_true, return_counts=True)
        outlier_label = unique_labels[np.argmin(counts)]
        y_binary = (y_true == outlier_label).astype(int)

        auc_results[ds] = {}
        for alg in algorithms:
            scores = load_scores(ds, alg, results_base_dir, nanread_base_dir)
            if scores is not None:
                fpr, tpr, _ = roc_curve(y_binary, scores)
                roc_auc = auc(fpr, tpr)
                if roc_auc < 0.5:
                    fpr, tpr, _ = roc_curve(y_binary, -scores)
                    roc_auc = auc(fpr, tpr)
                auc_results[ds][alg] = roc_auc
            else:
                auc_results[ds][alg] = np.nan

    df_auc = pd.DataFrame(auc_results).T
    df_auc.dropna(inplace=True) # Drop datasets with missing results

    if df_auc.empty:
        print("No complete data found to generate Nemenyi plot.")
        return

    num_datasets = len(df_auc)
    print(f"Generating Nemenyi diagram based on {num_datasets} datasets.")

    # Calculate ranks (descending: higher AUC gets rank 1)
    ranks_df = df_auc.rank(axis=1, ascending=False)
    avranks = ranks_df.mean().to_dict()

    # q_alpha values for alpha=0.05 (number of algorithms vs q_alpha)
    # Ref: Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets.
    # Nemenyi critical values for alpha=0.05:
    q_alpha_dict = {
        2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949,
        8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313
    }

    n_algs = len(algorithms)
    q_alpha = q_alpha_dict.get(n_algs, 3.219) # Default to 11 if not mapped

    cd = q_alpha * math.sqrt(n_algs * (n_algs + 1) / (6 * num_datasets))

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'Nemenyi_CD_diagram.png')

    plot_cd_diagram(avranks, num_datasets, cd, output_path)
    print(f"Successfully generated Nemenyi diagram at {output_path}")

if __name__ == '__main__':
    main()
