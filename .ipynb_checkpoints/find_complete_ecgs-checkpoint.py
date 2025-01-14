import os
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool
from tqdm import tqdm

# Define directories
input_dir = '/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-waveforms-npz'

# Function to process each file
def process_file(file_info):
    root, file = file_info

    if file.endswith('.npz'):
        file_path = os.path.join(root, file)
        try:
            arr = np.load(file_path)

            # Get the array name
            array_name = arr.files[0]
            data = arr[array_name]

            # Filter to 2.5s recordings
            data = data[:12]

            complete = 1  # Assume complete is 1

            for i in range(3):
                for lead in data[i::3]:
                    lead_without_nan = lead[~np.isnan(lead)]
                    if len(lead_without_nan) == 0:
                        complete = 0
                        break  # Exit inner loop
                if complete == 0:
                    break  # Exit outer loop

            # Get filename without extension
            filename = file[:-4]

            return (filename, complete)

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            filename = file[:-4]
            return (filename, -1)

def process_and_merge(df_path, df_complete, output_path):
    # Load dataframe
    df_to_merge = pd.read_csv(df_path)

    # Remove columns with '_x' or '_y' suffix
    cols_to_drop = [col for col in df_to_merge.columns if col.endswith('_x') or col.endswith('_y')]
    if cols_to_drop:
        print(f"Dropping columns from {df_path}: {cols_to_drop}")
        df_to_merge.drop(columns=cols_to_drop, inplace=True)

    # Remove 'complete' column if it exists
    if 'complete' in df_to_merge.columns:
        df_to_merge.drop(columns=['complete'], inplace=True)

    # Modify 'ecg_id_new' or create it
    # if 'ecg_id_new' in df_to_merge.columns:
    #     # Remove the last 4 characters from 'ecg_id_new'
    #     df_to_merge['ecg_id_new'] = df_to_merge['ecg_id_new'].astype(str).str[:-4]
    if 'ecg_id' in df_to_merge.columns:
        # Remove first 3 characters and last 4 characters from 'ecg_id' to create 'ecg_id_new'
        df_to_merge['ecg_id_new'] = df_to_merge['ecg_id'].astype(str).str[3:]
    else:
        raise ValueError(f"DataFrame from {df_path} does not have 'ecg_id_new' or 'ecg_id' column")

    # Merge on 'ecg_id_new'
    merged_df = df_to_merge.merge(df_complete, left_on='ecg_id_new', right_on='filename', how='left')

    # Drop 'filename' column after merge if not needed
    merged_df.drop(columns=['filename'], inplace=True)

    # Save the merged DataFrame
    merged_df.to_csv(output_path, index=False)
    print(f"Merged DataFrame saved to {output_path}")

    # Optional: Return merged DataFrame
    return merged_df

if __name__ == '__main__':
    # Gather all files
    all_files = [(root, file) for root, dirs, files in os.walk(input_dir) for file in files]

    # Use multiprocessing pool to parallelize the processing
    results = []
    with Pool() as pool:
        for res in tqdm(pool.imap(process_file, all_files), total=len(all_files)):
            results.append(res)

    # Create DataFrame from results
    df = pd.DataFrame(results, columns=['filename', 'complete'])

    # Remove entries with complete == -1 (errors)
    df = df[df['complete'] != -1]

    # List of DataFrames to process
    dataframes_to_merge = [
        "/home/ngsci/project/NEJM_benchmark/all_ids_labels_tested_with_covars_all.csv",
        "/home/ngsci/project/NEJM_benchmark/all_ids_labels_untested_with_covars_all.csv",
        "/home/ngsci/project/NEJM_benchmark/all_ids_labels_tested_with_covars.csv",
        "/home/ngsci/project/NEJM_benchmark/all_ids_labels_untested_with_covars.csv"
    ]

    # Process and merge each DataFrame
    for df_path in dataframes_to_merge:
        output_path = df_path  # Overwrite the original file
        process_and_merge(df_path, df, output_path)

    print("Processing complete.")

# if __name__ == '__main__':
#     # Gather all files
#     all_files = [(root, file) for root, dirs, files in os.walk(input_dir) for file in files]

#     # Use multiprocessing pool to parallelize the processing
#     results = []
#     with Pool() as pool:
#         for res in tqdm(pool.imap(process_file, all_files), total=len(all_files)):
#             results.append(res)

#     # Create dataframe from results
#     df = pd.DataFrame(results, columns=['ecg_id_new', 'complete'])

#     # Remove entries with complete == -1 (errors)
#     df = df[df['complete'] != -1]

#     # Load the first dataframe and merge
#     df1 = pd.read_csv("/home/ngsci/project/NEJM_benchmark/all_ids_labels_tested_with_covars_all.csv")
#     df1['ecg_id_new'] = df1['ecg_id'].astype(str).str[3:]
#     merged_df1 = df1.merge(df[['ecg_id_new', 'complete']], on='ecg_id_new', how='left')
#     merged_df1.to_csv("/home/ngsci/project/NEJM_benchmark/all_ids_labels_tested_with_covars_all.csv")
    
#     # Load the second dataframe and merge
#     df2 = pd.read_csv("/home/ngsci/project/NEJM_benchmark/all_ids_labels_untested_with_covars_all.csv")
#     merged_df2 = df2.merge(df[['ecg_id_new', 'complete']], on='ecg_id_new', how='left')
#     merged_df2.to_csv("/home/ngsci/project/NEJM_benchmark/all_ids_labels_untested_with_covars_all.csv")
    
#     # Optionally, save the merged dataframes
#     # merged_df1.to_csv('/path/to/save/merged_df1.csv', index=False)
#     # merged_df2.to_csv('/path/to/save/merged_df2.csv', index=False)

#     print("Processing complete.")
