import os
import numpy as np
import pandas as pd
import concurrent.futures
import hashlib
from functools import partial
from tqdm import tqdm

def compute_hash(file_path):
    """
    Compute a SHA256 hash for a given numpy array file.

    Parameters:
    - file_path (str): Path to the .npy file.

    Returns:
    - tuple: (file_id, hash_str)
    """
    try:
        array = np.load(file_path)
        # Ensure the array is in a consistent format
        array_bytes = array.tobytes()
        hash_str = hashlib.sha256(array_bytes).hexdigest()
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        return (file_id, hash_str)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return (None, None)

def find_duplicates(output_dir, output_csv='duplicates.csv', num_workers=None):
    """
    Find duplicate numpy arrays in the specified directory and save the results to a CSV.

    Parameters:
    - output_dir (str): Directory containing the .npy files.
    - output_csv (str): Filename for the output CSV.
    - num_workers (int, optional): Number of worker processes to use. Defaults to os.cpu_count().
    """
    # Gather all .npy file paths
    file_paths = [
        os.path.join(output_dir, fname)
        for fname in os.listdir(output_dir)
        if fname.endswith('.npy')
    ]

    print(f"Found {len(file_paths)} .npy files in {output_dir}.")

    # Use multiprocessing to compute hashes
    hash_dict = {}  # hash_str -> list of file_ids
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Use tqdm for progress bar
        results = list(tqdm(executor.map(compute_hash, file_paths), total=len(file_paths), desc="Computing hashes"))
    
    for file_id, hash_str in tqdm(results):
        if file_id is None or hash_str is None:
            continue  # Skip files that had errors
        if hash_str in hash_dict:
            hash_dict[hash_str].append(file_id)
        else:
            hash_dict[hash_str] = [file_id]
    
    print(f"Identified {len(hash_dict)} unique hash groups.")

    # Prepare data for DataFrame
    data = []
    for hash_str, file_ids in hash_dict.items():
        if len(file_ids) > 1:
            # All files in this group are duplicates of each other
            for fid in file_ids:
                duplicates = [dup_id for dup_id in file_ids if dup_id != fid]
                data.append({
                    'file_id': fid,
                    'duplicate_ids': duplicates,
                    'num_duplicates': len(duplicates)
                })
        else:
            # No duplicates
            fid = file_ids[0]
            data.append({
                'file_id': fid,
                'duplicate_ids': [],
                'num_duplicates': 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['file_id', 'duplicate_ids', 'num_duplicates'])

    # Optionally, sort the DataFrame by file_id
    df.sort_values(by='file_id', inplace=True)
    
    print(df.head(20))

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Duplicate information saved to {output_csv}.")

if __name__ == "__main__":
    # Define the output directory
    output_dir = '/home/ngsci/project/NEJM_benchmark/waveforms_3by4/'
    
    # Define the output CSV filename
    output_csv = 'ecg_duplicates.csv'
    
    # Call the function to find duplicates
    find_duplicates(output_dir, output_csv)
