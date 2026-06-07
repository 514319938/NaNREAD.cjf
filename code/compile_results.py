import pandas as pd
import os
import sys

def main():
    results_dir = 'results'
    base_file = os.path.join(results_dir, 'all_outlier_result-20251025.xls')
    output_file = os.path.join(results_dir, 'all_outlier_result.xls')

    print(f"Reading base results from {base_file}...")
    try:
        df = pd.read_excel(base_file)
    except Exception as e:
        print(f"Error reading base file: {e}")
        sys.exit(1)

    # Dictionary to hold the NaNREAD AUC results mapping dataset -> AUC
    nanread_results = {}

    # Iterate over directories in results/
    if os.path.exists(results_dir):
        for entry in os.listdir(results_dir):
            entry_path = os.path.join(results_dir, entry)
            if os.path.isdir(entry_path):
                dataset_name = entry
                xls_path = os.path.join(entry_path, f"{dataset_name}.xls")
                if os.path.exists(xls_path):
                    try:
                        # Ensure we read it correctly, first row, column 'opt__ROC__AUC'
                        ds_df = pd.read_excel(xls_path)
                        if 'opt__ROC__AUC' in ds_df.columns:
                            auc = ds_df['opt__ROC__AUC'].iloc[0]
                            nanread_results[dataset_name] = auc
                    except Exception as e:
                        print(f"Error reading {xls_path}: {e}")

    # Add the NaNREAD column to the dataframe
    df['NaNREAD'] = df['dataset'].map(nanread_results)

    # Remove the unnamed columns if any
    cols_to_drop = [c for c in df.columns if str(c).startswith('Unnamed:')]
    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    print(f"Collected NaNREAD results for {len(nanread_results)} datasets.")

    # Save the merged results
    try:
        # use openpyxl as engine for writing to keep it consistent
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"Successfully saved compiled results to {output_file}")
    except Exception as e:
        print(f"Failed to save {output_file}: {e}")

if __name__ == '__main__':
    main()
