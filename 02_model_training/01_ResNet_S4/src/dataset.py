import lightning.pytorch as pl
import torchvision
import torch
import torchio as tio
import math
import numpy as np
import pandas as pd
import json
import tqdm
import os
from collections import Counter
import pickle
from scipy.signal import butter, filtfilt
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from PIL import Image


##########################################
######## ECG Sequence Datasets ###########
##########################################
    
class ECG_Seq(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for ECG waveform sequence dataset. This will load the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, 
                 use_data_augmentation=False, 
                 batch_size=64, 
                 num_workers=8,
                 img_size = [256, 256],
                 num_images = 1,
                 label_file = "/home/ngsci/project/ACS_benchmark/data/all_ids_labels_tested_with_covars_all_final_cath.csv", 
                 label_key = "stent_or_cabg_010_day",
                 class_balance=False,
                 transform= 'bandpass', 
                 covars = [],
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.use_data_augmentation = use_data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_images = num_images
        self.img_size = img_size
        self.class_balance = class_balance
        self.transform = transform
        self.covars = covars
        
        #Initialize the dataset file
        self.label_file = pd.read_csv(label_file)
        self.label_file['y'] = self.label_file[label_key]

        self.prepare_data()
    
    def prepare_data(self):
        
        self.train_dataset = self.label_file[self.label_file["split"]=="train"].reset_index()
        self.val_dataset = self.label_file[self.label_file["split"]=="valid"].reset_index()
        self.test_dataset = self.label_file[self.label_file["split"]=="test"].reset_index()

        # Initialize scaler and KNN imputer
        if len(self.covars) > 0:
            self.covars_scaler = StandardScaler()
            
            if 'maxtrop_sameday' in self.covars:
                
                # Initialize the SimpleImputer with median strategy
                self.covars_imputer = SimpleImputer(strategy='median')

                # Fit the imputer on the training set and transform/impute missing values
                self.train_dataset['maxtrop_sameday'] = self.covars_imputer.fit_transform(self.train_dataset[['maxtrop_sameday']])

                # Impute missing values in the validation and test sets using the same imputer
                # Note: Here we use just `.transform()` instead of `.fit_transform()` because we want to use the median calculated from the training set
                self.val_dataset['maxtrop_sameday'] = self.covars_imputer.transform(self.val_dataset[['maxtrop_sameday']])
                self.test_dataset['maxtrop_sameday'] = self.covars_imputer.transform(self.test_dataset[['maxtrop_sameday']])
                
                mean_trop = self.train_dataset['maxtrop_sameday'].mean()
                std_trop = self.train_dataset['maxtrop_sameday'].std()
                self.train_dataset['maxtrop_sameday'] = (self.train_dataset['maxtrop_sameday'] - mean_trop) / std_trop
                self.val_dataset['maxtrop_sameday'] = (self.val_dataset['maxtrop_sameday'] - mean_trop) / std_trop
                self.test_dataset['maxtrop_sameday'] = (self.test_dataset['maxtrop_sameday'] - mean_trop) / std_trop
                
            # Now fit the scaler on the imputed training set
            self.covars_scaler.fit(self.train_dataset[self.covars])
        else:
            self.covars_scaler = None
            self.covars_imputer = None

    def setup(self, stage=None):
        
        self.train_dataset = self.label_file[self.label_file["split"]=="train"].reset_index()
        self.val_dataset = self.label_file[self.label_file["split"]=="valid"].reset_index()
        self.test_dataset = self.label_file[self.label_file["split"]=="test"].reset_index()
        
        if self.class_balance:
            if self.class_balance:
                y_train_sampler = np.array(self.train_dataset['y'])
            
            y_train_sampler_unique = np.unique(y_train_sampler)

            class_sample_count = np.array([len(np.where(y_train_sampler==t)[0]) for t in y_train_sampler_unique])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[list(y_train_sampler_unique).index(t)] for t in y_train_sampler])

            samples_weight = torch.from_numpy(samples_weight)
            self.train_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        
        self.train = ECGSeq_Dataset(dataset=self.train_dataset, covars=self.covars, scaler=self.covars_scaler, transform=self.transform)
        self.val = ECGSeq_Dataset(dataset=self.val_dataset,  covars=self.covars, scaler=self.covars_scaler, transform=self.transform)
        self.test = ECGSeq_Dataset(dataset=self.test_dataset,  covars=self.covars, scaler=self.covars_scaler, transform=self.transform)

    def train_dataloader(self):
        if self.class_balance:
            return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.train_sampler)
        else:
            return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class ECGSeq_Dataset(torch.utils.data.Dataset):
    """
    Pytorch Dataset for 1D ECG representation dataset. Loads ECGs and attaches label.
    """

    def __init__(self, dataset, covars=[], scaler=None, transform='butterworth', fs=500.0):
        """
        Initialize the dataset.
        
        Args:
        - dataset: DataFrame containing the dataset information.
        - covars: List of covariate names to include in each sample.
        - scaler: Scaler object for normalizing covariates.
        - transform: Type of transform to apply to ECG signals, options are 'bandpass', 'butterworth', or None.
        - fs: Sampling frequency of the ECG signals (default 500 Hz).
        """
        self.dataset = dataset
        self.covars = covars
        self.covars_scaler = scaler
        self.transform = transform
        self.fs = fs  # Sampling frequency

        print(self.get_summary_statement())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ecgid = self.dataset.iloc[idx]['ecg_id_new'].split(".")[0]
        sample_path = f"/home/ngsci/project/ACS_benchmark/waveforms_3by4/{ecgid}.npy" 
        sample = {}
        sample["x"] = torch.Tensor(np.load(sample_path)).float()

        # Apply the selected transformation
        if self.transform == 'bandpass':
            sample["x"] = self.apply_bandpass_filter(sample["x"].numpy())
        elif self.transform == 'butterworth':
            sample["x"] = self.apply_butterworth_filter_4poles(sample["x"].numpy())

        if len(self.covars) > 0:
            covars_data = self.dataset.iloc[idx][self.covars]
            # Normalize the covariates
            covars_data = self.covars_scaler.transform([covars_data])
            covars_data = covars_data.flatten()
            sample['cov'] = covars_data

        sample["y"] = self.dataset.iloc[idx]['y']
        sample['id'] = ecgid

        return sample

    def apply_bandpass_filter(self, ecg_signal):
        """
        Apply a generic bandpass filter with user-defined parameters.
        Uses a Butterworth filter.
        
        Args:
        - ecg_signal: Numpy array of ECG signal to filter.
        
        Returns:
        - Filtered ECG signal as a torch Tensor.
        """
        lowcut = 0.5  # Define default lowcut frequency
        highcut = 50.0  # Define default highcut frequency
        order = 5  # Filter order
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        filtered_signal = torch.Tensor(filtered_signal.copy())
        return filtered_signal #torch.Tensor(filtered_signal)

    def apply_butterworth_filter_4poles(self, ecg_signal):
        """
        Apply a Butterworth bandpass filter with 4 poles (order = 4), keeping information between 0.2 Hz and 25 Hz.
        
        Args:
        - ecg_signal: Numpy array of ECG signal to filter.
        
        Returns:
        - Filtered ECG signal as a torch Tensor.
        """
        lowcut = 0.2  # 0.2 Hz lower cutoff
        highcut = 25.0  # 25 Hz upper cutoff
        order = 4  # 4 poles means order=4
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)
        filtered_signal = torch.Tensor(filtered_signal.copy())
        return filtered_signal

    def get_summary_statement(self):
        """
        Generate a summary statement of the dataset.
        
        Returns:
        - Summary string of the dataset.
        """
        num_outcome = sum(self.dataset['y'])
        return "ECG Dataset. {} ECGs ({} with outcome)".format(len(self.dataset), num_outcome)
    