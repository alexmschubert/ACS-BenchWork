import os
import numpy as np
import biosppy
from scipy import stats
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from tqdm import tqdm

# Define directories
input_dir = '/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-waveforms-npz'
output_dir = '/home/ngsci/project/NEJM_benchmark/waveforms_3by4/'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define the number of samples to take around each R-peak
samples_around_r_peak = 250  # 250 samples before and after the R-peak, total 500

# Function to process each file
def process_file(file_info):
    root, file = file_info
    global lost_files

    if file.endswith('.npz'):
        file_path = os.path.join(root, file)
        try:
            arr = np.load(file_path)

            # Get the array name
            array_name = arr.files[0]
            data = arr[array_name]

            # Filter to 2.5s recordings
            data = data[:12]

            # Initialize the output array
            ecg = np.zeros((3, 5000))

            for i in range(3):
                for lead in data[i::3]:
                    # Subtract the mode
                    lead_without_nan = lead[~np.isnan(lead)]
                    if len(lead_without_nan)==0:
                        mode_value = 0
                    else:
                        mode_value = stats.mode(lead_without_nan)[0][0]
                    lead = lead - mode_value
                    
                    # Add lead to ECG
                    lead = np.nan_to_num(lead, nan=0)
                    lead = lead[:5000]
                    ecg[i] += lead

            # Save the median beats array to the output directory
            filename = file[:-4]
            output_path = os.path.join(output_dir, f'{filename}.npy')
            np.save(output_path, ecg)

        except Exception as e:
            # with lost_files.get_lock():
            #     lost_files.value += 1
            lost_files.append(file)
            print(f"Error processing file {file}: {e}")

# Main script
if __name__ == '__main__':
    manager = Manager()
    #lost_files = manager.Value('i', 0)
    lost_files = manager.list()  

    # Gather all files
    all_files = [(root, file) for root, dirs, files in os.walk(input_dir) for file in files]

    # Use multiprocessing pool to parallelize the processing
    with Pool() as pool:
        list(tqdm(pool.imap(process_file, all_files), total=len(all_files)))

    print(f'{len(lost_files)} files could not be processed')
    print("All files processed.")
