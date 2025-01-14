import os
import numpy as np
from biosppy.signals import ecg as bioecg
import scipy.signal
import multiprocessing
from functools import partial
from tqdm import tqdm

def process_ecg_file(ecg_file, output_dir):
    """
    Processes an ECG .npz file to create a 12-lead, 10-second ECG signal. 
    For all 2.5 second beats we compute the median beat and 

    Args:
        ecg_file (str): Path to the input ECG .npz file.
        output_dir (str): Path to the output directory where the processed ECG will be saved.

    Returns:
        None
    """
    try:
        # Load the npz file
        arr = np.load(ecg_file)
        
        # Access the ECG data array
        array_name = arr.files[0]
        data = arr[array_name]
        
        # Define sampling rate and time vector
        sampling_rate = 500  # Hz
        total_samples = data.shape[1]  # Should be 5000 samples (10 seconds at 500 Hz)
        time_vector = np.arange(total_samples) / sampling_rate
        
        # Extract R-peak locations from lead II (data[13])
        lead_ii = data[13]
        lead_ii_clean = np.nan_to_num(lead_ii, nan=0.0)
        
        # R-peak detection using biosppy
        r_peak_results = bioecg.hamilton_segmenter(signal=lead_ii_clean, sampling_rate=sampling_rate)
        r_peaks = r_peak_results['rpeaks']
        
        # Initialize an array to hold the processed leads
        processed_leads = np.zeros((12, total_samples))
        
        for lead_idx in range(12):
            lead_data = data[lead_idx]
            
            # Drop NaNs and get valid data (first 2.5 seconds)
            valid_indices = ~np.isnan(lead_data)
            lead_data_no_nan = lead_data[valid_indices]
            
            # Check if there's enough data
            if len(lead_data_no_nan) == 0:
                continue  
            
            # Adjust R-peaks to indices within valid data
            max_valid_index = valid_indices.nonzero()[0][-1]
            min_valid_index = valid_indices.nonzero()[0][0]
            
            # R-peaks within the valid data range
            r_peaks_valid = r_peaks[(r_peaks >= min_valid_index) & (r_peaks <= max_valid_index)]
            r_peaks_adjusted = r_peaks_valid - min_valid_index 
            
            # Check if there are enough R-peaks
            if len(r_peaks_adjusted) < 1:
                continue  
            
            # Extract beats from valid data
            beats = []
            for r_peak in r_peaks_adjusted:
                # Define a window around the R-peak
                window_size = int(0.8 * sampling_rate)  # 0.8 seconds window
                half_window = window_size // 2
                start = int(r_peak) - half_window
                end = int(r_peak) + half_window
                
                # Ensure indices are within bounds
                if start < 0 or end > len(lead_data_no_nan):
                    continue
                beat = lead_data_no_nan[start:end]
                beats.append(beat)
            
            if len(beats) == 0:
                continue  
            
            # Align beats by truncating to the shortest length
            beat_length = min(len(beat) for beat in beats)
            beats_aligned = [beat[:beat_length] for beat in beats]
            
            # Compute median beat
            median_beat = np.median(np.array(beats_aligned), axis=0)
            
            # Apply window function to smooth edges
            window = np.hanning(len(median_beat))
            median_beat_windowed = median_beat * window
            
            # Initialize the extended lead signal with zeros
            extended_lead = np.zeros(total_samples)
            
            # Length of the median beat
            beat_len = len(median_beat_windowed)
            half_beat_len = beat_len // 2
            
            # Place median beats at R-peak locations from lead II over the full 10 seconds
            for r_peak in r_peaks:
                start_idx = int(r_peak) - half_beat_len
                end_idx = int(r_peak) + half_beat_len
                
                # Adjust indices if they go beyond the signal boundaries
                beat_start = 0
                beat_end = beat_len
                if start_idx < 0:
                    beat_start = -start_idx
                    start_idx = 0
                if end_idx > total_samples:
                    beat_end = beat_len - (end_idx - total_samples)
                    end_idx = total_samples
                
                if beat_start >= beat_end:
                    continue
                
                # Add the median beat to the extended lead
                extended_lead[start_idx:end_idx] += median_beat_windowed[beat_start:beat_end]
            
            # Store the processed lead
            processed_leads[lead_idx] = extended_lead
        
        # Include the 3 leads that were recorded for 10 seconds as is
        last_3_leads = data[12:15]
        lead_locations = [6, 1, 10]  
        
        for i in range(3):
            location_ = lead_locations[i]
            lead = last_3_leads[i]
            lead = np.nan_to_num(lead, nan=0.0)
            
            processed_leads[location_] = lead
        
        # Save the processed ECG
        base_filename = os.path.basename(ecg_file)
        filename_no_ext = os.path.splitext(base_filename)[0]
        output_file = os.path.join(output_dir, filename_no_ext + '.npy')
        np.save(output_file, processed_leads)
        
        return True
        
    except Exception as e:
        print(f"Error processing {ecg_file}: {e}")
        return False

def main():
    input_dir = '/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-waveforms-npz/'
    output_dir = '/home/ngsci/project/ACS_benchmark/waveforms_12lead_10sec/'
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all .npz files in the input directory and subdirectories
    ecg_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.npz'):
                file_path = os.path.join(root, file)
                ecg_files.append(file_path)
    
    print(f"Found {len(ecg_files)} ECG files to process.")
    
    # Use multiprocessing to process files in parallel
    num_workers = multiprocessing.cpu_count()-1  # Use the number of available CPU cores
    print(f"Using {num_workers} worker processes.")
    
    # Create a partial function to fix the output_dir parameter
    process_func = partial(process_ecg_file, output_dir=output_dir)
    
    # Use multiprocessing for speed-up
    with multiprocessing.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap_unordered(process_func, ecg_files), total=len(ecg_files)))
    
    # Check for any failures
    num_success = sum(results)
    num_failures = len(results) - num_success
    print(f"Processing complete. Successful: {num_success}, Failed: {num_failures}")

if __name__ == "__main__":
    main()
