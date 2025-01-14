import os
import numpy as np
import torch
from scipy import signal
from scipy.ndimage import median_filter
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm
import argparse
    
def apply_notch_filter(data, sampling_frequency=500):
    """Applies a notch filter to remove powerline interference."""
    row, __ = data.shape
    processed_data = np.zeros(data.shape)
    b = np.ones(int(0.02 * sampling_frequency)) / 50.0
    a = [1]
    for lead in range(row):
        X = signal.filtfilt(b, a, data[lead, :])
        processed_data[lead, :] = X
    return processed_data 


def apply_baseline_filter(data, sampling_frequency=500):
    """Applies a baseline correction filter to remove low-frequency drift."""
    row, __ = data.shape
    
    win_size = int(np.round(0.2 * sampling_frequency)) + 1
    baseline = median_filter(data, [1, win_size], mode="constant")
    win_size = int(np.round(0.6 * sampling_frequency)) + 1
    baseline = median_filter(baseline, [1, win_size], mode="constant")
    filt_data = data - baseline
    return filt_data


def process_ecg_file(input_file, output_dir):
    """Loads, processes, and saves the ECG file."""
    try:
        # Load ECG data
        data = np.load(input_file).astype(np.float32)

        # Apply filters
        data = apply_notch_filter(data)
        data = apply_baseline_filter(data)

        # Save processed ECG
        output_file = os.path.join(output_dir, os.path.basename(input_file))
        np.save(output_file, data)
    except Exception as e:
        print(f"Error processing {input_file}: {e}")
    
#     # Load ECG data
#     data = np.load(input_file).astype(np.float32)

#     # Apply filters
#     data = apply_notch_filter(data)
#     data = apply_baseline_filter(data)

#     # Save processed ECG
#     output_file = os.path.join(output_dir, os.path.basename(input_file))
#     np.save(output_file, data)


def main(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Gather all input ECG files
    ecg_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".npy")]

    # Use multiprocessing to process ECG files
    with Pool(cpu_count()) as pool:
        list(tqdm.tqdm(pool.imap(partial(process_ecg_file, output_dir=output_dir), ecg_files), total=len(ecg_files)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process ECG files with notch and baseline filters.")
    parser.add_argument("--input_dir", 
                        type=str,
                        default='/home/ngsci/project/NEJM_benchmark/waveforms_12lead_10sec',
                        help="Directory containing input ECG files (.npy format).")
    parser.add_argument("--output_dir", 
                        type=str,
                        default='/home/ngsci/project/NEJM_benchmark/waveforms_12lead_10sec_baseline_notch',
                        help="Directory to save processed ECG files.")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
