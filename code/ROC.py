import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def load_ground_truth(dataset_name, data_dir='data'):
    """Load ground truth labels from the original dataset .mat file."""
    mat_path = os.path.join(data_dir, f"{dataset_name}.mat")
    if not os.path.exists(mat_path):
        print(f"Warning: Data file {mat_path} not found.")
        return None

    try:
        data_dict = sio.loadmat(mat_path)
        if 'trandata' in data_dict:
            data = data_dict['trandata']
            y_true = data[:, -1]
            return y_true
        else:
            print(f"Warning: 'trandata' not found in {mat_path}")
            return None
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

def load_scores(dataset_name, alg, results_base_dir, nanread_base_dir):
    """Load outlier scores for a specific algorithm and dataset."""
    if alg == 'NaNREAD':
        # NaNREAD results are stored in the local results directory
        mat_path = os.path.join(nanread_base_dir, dataset_name, f"{dataset_name}.mat")
    else:
        # Baseline algorithms are in F:\Experiental_results
        alg_dir = f"{alg}_results"
        mat_path = os.path.join(results_base_dir, alg_dir, dataset_name, f"{dataset_name}_{alg}.mat")

    if not os.path.exists(mat_path):
        print(f"Warning: Result file {mat_path} not found for {alg}.")
        return None

    try:
        mat_data = sio.loadmat(mat_path)
        # Handle variations in variable names
        if 'opt_out_scores' in mat_data:
            scores = mat_data['opt_out_scores'].flatten()
        elif 'opt__out__scores' in mat_data:
            scores = mat_data['opt__out__scores'].flatten()
        else:
            print(f"Warning: Score variable not found in {mat_path}")
            return None
        return scores
    except Exception as e:
        print(f"Error loading {mat_path}: {e}")
        return None

def plot_roc_for_dataset(dataset_name, algorithms, data_dir, results_base_dir, nanread_base_dir, output_dir):
    """Generate and save ROC curve for a single dataset."""
    y_true = load_ground_truth(dataset_name, data_dir)
    if y_true is None:
        return

    plt.figure(figsize=(10, 8))

    # Store AUCs for legend sorting
    auc_dict = {}
    roc_data = {}

    for alg in algorithms:
        scores = load_scores(dataset_name, alg, results_base_dir, nanread_base_dir)
        if scores is not None:
            # check if scores are inversely correlated with outliers
            try:
                fpr, tpr, _ = roc_curve(y_true, scores)
                roc_auc = auc(fpr, tpr)
                if roc_auc < 0.5:
                    fpr, tpr, _ = roc_curve(y_true, -scores)
                    roc_auc = auc(fpr, tpr)

                roc_data[alg] = (fpr, tpr, roc_auc)
                auc_dict[alg] = roc_auc
            except Exception as e:
                print(f"Error calculating ROC for {alg} on {dataset_name}: {e}")

    if not roc_data:
        print(f"No valid scores found for {dataset_name}. Skipping plot.")
        plt.close()
        return

    # Sort algorithms by AUC in descending order for plotting
    sorted_algs = sorted(auc_dict.keys(), key=lambda x: auc_dict[x], reverse=True)

    for alg in sorted_algs:
        fpr, tpr, roc_auc = roc_data[alg]
        # Make NaNREAD stand out
        if alg == 'NaNREAD':
            plt.plot(fpr, tpr, label=f'{alg} (AUC = {roc_auc:.4f})', linewidth=2.5, zorder=10)
        else:
            plt.plot(fpr, tpr, label=f'{alg} (AUC = {roc_auc:.4f})', linewidth=1.5)

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curves - {dataset_name}', fontsize=16)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{dataset_name}_ROC.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve for {dataset_name} to {output_path}")

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

    # Configure directories - using relative paths so it works anywhere
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    nanread_base_dir = os.path.join(base_dir, 'results')
    output_dir = os.path.join(base_dir, 'results_figures')

    # For local execution, results_base_dir points to the F: drive
    # If using Linux/Mac, adjust this path. If running on Windows, 'F:\\Experiental_results' works.
    if os.name == 'nt': # Windows
        results_base_dir = r'F:\Experiental_results'
    else: # Provide a placeholder for testing on non-Windows
        results_base_dir = '/tmp/Experiental_results'

    print(f"Using results directory: {results_base_dir}")
    print(f"Starting ROC curve generation for {len(datasets)} datasets...")
    for ds in datasets:
        print(f"\nProcessing dataset: {ds}")
        plot_roc_for_dataset(ds, algorithms, data_dir, results_base_dir, nanread_base_dir, output_dir)
    print("\nDone!")

if __name__ == '__main__':
    main()
