import lightning.pytorch as pl
import torchvision
import medmnist
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


CACHE_IMG_SIZE = [256, 256]

class ECG_IMG(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for ECG waveform image dataset. This will load the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, 
                 use_data_augmentation=False, 
                 batch_size=32, 
                 num_workers=32,
                 img_size = [256, 256],
                 num_images = 1,
                 label_file = "/home/ngsci/project/NEJM_benchmark/all_ids_labels_tested.csv", #"/home/ngsci/project/NEJM_benchmark/all_ids_labels_untested.csv",
                 label_key = "has_st_eleva",
                 class_balance=True,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.use_data_augmentation = use_data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_images = num_images
        self.img_size = img_size
        self.class_balance = class_balance
        
        #Initialize the dataset file
        self.label_file = pd.read_csv(label_file)
        self.label_file['y'] = self.label_file[label_key]

        self.prepare_data_transforms()

    def prepare_data_transforms(self):
        '''
            Prepare data transforms for train and test data.
            Note, you may want to apply data augmentation (see torchvision) for the train data.
        '''
        
        self.test_transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # torchvision.transforms.CenterCrop(224),
            # torchvision.transforms.ToTensor(),  # Converts PIL Image to Tensor and scales to [0, 1]
            #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor()
        ])
        if self.use_data_augmentation:
            # TODO: Implement some data augmentatons
            self.train_transform = torchvision.transforms.Compose([
                            torchvision.transforms.Resize((256, 256)),#torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                            #torchvision.transforms.CenterCrop(224),
                
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomVerticalFlip(),
                            #torchvision.transforms.RandomRotation(10),
                            torchvision.transforms.RandomGrayscale(0.3),
                            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
                            # torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
                            torchvision.transforms.ToTensor(),  
                
                            #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  
                        ])
        else:
            self.train_transform = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                # torchvision.transforms.CenterCrop(224),
                # torchvision.transforms.ToTensor(),  # Converts PIL Image to Tensor and scales to [0, 1]
                #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor()
            ])

    def prepare_data(self):
        
        self.train_dataset = self.label_file[self.label_file["split"]=="train"]
        self.val_dataset = self.label_file[self.label_file["split"]=="valid"]
        self.test_dataset = self.label_file[self.label_file["split"]=="test"]

    def setup(self, stage=None):
        
        self.train_dataset = self.label_file[self.label_file["split"]=="train"].reset_index()
        self.val_dataset = self.label_file[self.label_file["split"]=="valid"].reset_index()
        self.test_dataset = self.label_file[self.label_file["split"]=="test"].reset_index()
        
        if self.class_balance:
            # This data is highly imbalanced!
            # Introduce a method to deal with class imbalance (hint: think about your data loader)
            if self.class_balance:
                y_train_sampler = np.array(self.train_dataset['y'])
            
            y_train_sampler_unique = np.unique(y_train_sampler)

            class_sample_count = np.array([len(np.where(y_train_sampler==t)[0]) for t in y_train_sampler_unique])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[list(y_train_sampler_unique).index(t)] for t in y_train_sampler])

            samples_weight = torch.from_numpy(samples_weight)
            self.train_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        
        # TODO Here we should initialize the splitted train, val and test sets based on the labelfile data
        self.train = IMG_Dataset(dataset=self.train_dataset, transforms = self.train_transform)
        self.val = IMG_Dataset(dataset=self.val_dataset, transforms = self.test_transform)
        self.test = IMG_Dataset(dataset=self.test_dataset, transforms = self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class IMG_Dataset(torch.utils.data.Dataset):
    """
        Pytorch Dataset for NLST dataset. Loads preprocesses data from disk and applies data augmentation. Generates masks from bounding boxes stored in metadata..
    """

    def __init__(self, dataset, transforms, img_size=[256, 256], num_images=1, as_rgb=True): #normalize,
        self.dataset = dataset
        self.transform = transforms
        #self.normalize = normalize
        self.img_size = img_size
        self.num_images = num_images
        self.as_rgb = as_rgb

        print(self.get_summary_statement())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ecgid = self.dataset.iloc[idx]['ecg_id'].split(".")[0]
        sample_path = f"/home/ngsci/project/NEJM_benchmark/waveforms_img/{ecgid}.png"
        sample = {}
        sample["x"] = Image.open(sample_path)

        if self.as_rgb:
            sample["x"] = sample["x"].convert('RGB')

        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])

        sample["y"] = self.dataset.iloc[idx]['y']

        return sample

    def get_summary_statement(self):
        num_patients = len(self.dataset['ecg_id']) #len(set([d['ecgid'] for d in self.dataset]))
        num_outcome = sum(self.dataset['y']) #sum([d['y'] for d in self.dataset])
        return "ECG Dataset. {} ECGs ({} with outcome)".format(len(self.dataset),  num_outcome)

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
                 label_file = "/home/ngsci/project/NEJM_benchmark/all_ids_labels_tested_with_covars_all.csv", #"/home/ngsci/project/NEJM_benchmark/all_ids_labels_NAIVE.csv" #"/home/ngsci/project/NEJM_benchmark/all_ids_labels_untested_with_covars.csv",
                 label_key = "stent_or_cabg_010_day",
                 class_balance=False,
                 transform= 'bandpass', #'butterworth',
                 only_complete = False,
                 drop_duplicates = False,
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
        # if len(covars) > 0:
        #     covars_list = json.loads(covars)
        #     self.covars = covars_list 
        # else:
        self.covars = covars
        
        #Initialize the dataset file
        self.label_file = pd.read_csv(label_file)
        self.label_file['y'] = self.label_file[label_key]
        
        
        print(self.label_file.columns)
        if only_complete:
            print(self.label_file[['ecg_id_new', 'complete']].head())
            self.label_file = self.label_file[self.label_file['complete']==1]
            print(self.label_file.shape)
        
        if drop_duplicates:
            print(self.label_file[['ecg_id_new', 'num_duplicates']].head())
            self.label_file = self.label_file[self.label_file['num_duplicates']<1]
            print(self.label_file.shape)

        #self.prepare_data_transforms()
        self.prepare_data()
    
    def prepare_data(self):
        
        self.train_dataset = self.label_file[self.label_file["split"]=="train"].reset_index()
        self.val_dataset = self.label_file[self.label_file["split"]=="valid"].reset_index()
        self.test_dataset = self.label_file[self.label_file["split"]=="test"].reset_index()

        # Initialize scaler and KNN imputer
        if len(self.covars) > 0:
            self.covars_scaler = StandardScaler()
            
            if 'maxtrop_sameday' in self.covars:
#                 # Initialize the KNNImputer
#                 self.covars_imputer = KNNImputer(n_neighbors=5)

#                 # Fit the imputer on the training set
#                 self.covars_imputer.fit_transform(self.train_dataset[['maxtrop_sameday']])

#                 # Impute missing values in train, val, and test sets
#                 #self.train_dataset['maxtrop_sameday'] = self.covars_imputer.transform(self.train_dataset[['maxtrop_sameday']])
#                 self.val_dataset['maxtrop_sameday'] = self.covars_imputer.transform(self.val_dataset[['maxtrop_sameday']])
#                 self.test_dataset['maxtrop_sameday'] = self.covars_imputer.transform(self.test_dataset[['maxtrop_sameday']])
                
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
        
        # output_dir = "/home/ngsci/project/NEJM_benchmark/waveforms_med_beat/"
        # # List all files in the input directory
        # ecg_files = os.listdir(output_dir)
        # self.label_file = self.label_file[self.label_file["ecg_id"].isin(ecg_files)].reset_index()
        
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
        
        # TODO Consider enabling 1D transforms
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
        sample_path = f"/home/ngsci/project/NEJM_benchmark/waveforms_3by4/{ecgid}.npy" 
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
        lowcut = 0.5  # Define your default lowcut frequency
        highcut = 50.0  # Define your default highcut frequency
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
        num_patients = len(self.dataset['ecg_id_new'])
        num_outcome = sum(self.dataset['y'])
        return "ECG Dataset. {} ECGs ({} with outcome)".format(len(self.dataset), num_outcome)
    
# class ECGSeq_Dataset(torch.utils.data.Dataset):
#     """
#         Pytorch Dataset for 1D ECG representation dataset. Loads ECGs and attaches label.
#     """

#     def __init__(self, dataset, covars=[], scaler=None): #normalize,
#         self.dataset = dataset
#         self.covars = covars
#         self.covars_scaler = scaler
             
#         print(self.get_summary_statement())

#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, idx):
#         ecgid = self.dataset.iloc[idx]['ecg_id_new'].split(".")[0]
#         sample_path = f"/home/ngsci/project/NEJM_benchmark/waveforms_3by4/{ecgid}.npy"
#         sample = {}
#         #sample["x"] = Image.open(sample_path)
#         sample["x"] = torch.Tensor(np.load(sample_path)).float()
        
#         if len(self.covars)>0: 
#             covars_data = self.dataset.iloc[idx][self.covars]
#             # Normalize the covariates
#             covars_data = self.covars_scaler.transform([covars_data])
#             covars_data = covars_data.flatten()
#             sample['cov'] = covars_data

#         sample["y"] = self.dataset.iloc[idx]['y']
#         sample['id'] = ecgid

#         return sample

#     def get_summary_statement(self):
#         num_patients = len(self.dataset['ecg_id_new']) #len(set([d['ecgid'] for d in self.dataset]))
#         num_outcome = sum(self.dataset['y']) #sum([d['y'] for d in self.dataset])
#         return "ECG Dataset. {} ECGs ({} with outcome)".format(len(self.dataset),  num_outcome)


#####################################
######## 1D Conv Datasets ###########
#####################################

class ECG_1D(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for ECG waveform image dataset. This will load the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, 
                 use_data_augmentation=False, 
                 batch_size=32, 
                 num_workers=32,
                 img_size = [256, 256],
                 num_images = 1,
                 label_file = "/home/ngsci/project/NEJM_benchmark/all_ids_labels_tested_with_covars.csv", #"/home/ngsci/project/NEJM_benchmark/all_ids_labels_untested_with_covars.csv",
                 label_key = "has_st_eleva",
                 class_balance=True,
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
        # if len(covars) > 0:
        #     covars_list = json.loads(covars)
        #     self.covars = covars_list 
        # else:
        self.covars = covars
        
        #Initialize the dataset file
        self.label_file = pd.read_csv(label_file)
        self.label_file['y'] = self.label_file[label_key]

        #self.prepare_data_transforms()
        self.prepare_data()
    
    #TODO adapt this to 1D data
    def prepare_data_transforms(self):
        '''
            Prepare data transforms for train and test data.
            Note, you may want to apply data augmentation (see torchvision) for the train data.
        '''
        
        self.test_transform = torchvision.transforms.Compose([
            # torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            # torchvision.transforms.CenterCrop(224),
            # torchvision.transforms.ToTensor(),  # Converts PIL Image to Tensor and scales to [0, 1]
            #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor()
        ])
        if self.use_data_augmentation:
            # TODO: Implement some data augmentatons
            self.train_transform = torchvision.transforms.Compose([
                            torchvision.transforms.Resize((256, 256)),#torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                            #torchvision.transforms.CenterCrop(224),
                
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomVerticalFlip(),
                            #torchvision.transforms.RandomRotation(10),
                            torchvision.transforms.RandomGrayscale(0.3),
                            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
                            # torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
                            torchvision.transforms.ToTensor(),  
                
                            #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  
                        ])
        else:
            self.train_transform = torchvision.transforms.Compose([
                # torchvision.transforms.Resize(256, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                # torchvision.transforms.CenterCrop(224),
                # torchvision.transforms.ToTensor(),  # Converts PIL Image to Tensor and scales to [0, 1]
                #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                torchvision.transforms.Resize((256, 256)),
                torchvision.transforms.ToTensor()
            ])
    
    def prepare_data(self):
        
        self.train_dataset = self.label_file[self.label_file["split"]=="train"]
        self.val_dataset = self.label_file[self.label_file["split"]=="valid"]
        self.test_dataset = self.label_file[self.label_file["split"]=="test"]

        # Initialize scaler and KNN imputer
        if len(self.covars) > 0:
            self.covars_scaler = StandardScaler()
            
            if 'maxtrop_sameday' in self.covars:
#                 # Initialize the KNNImputer
#                 self.covars_imputer = KNNImputer(n_neighbors=5)

#                 # Fit the imputer on the training set
#                 self.covars_imputer.fit_transform(self.train_dataset[['maxtrop_sameday']])

#                 # Impute missing values in train, val, and test sets
#                 #self.train_dataset['maxtrop_sameday'] = self.covars_imputer.transform(self.train_dataset[['maxtrop_sameday']])
#                 self.val_dataset['maxtrop_sameday'] = self.covars_imputer.transform(self.val_dataset[['maxtrop_sameday']])
#                 self.test_dataset['maxtrop_sameday'] = self.covars_imputer.transform(self.test_dataset[['maxtrop_sameday']])
                
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
        
        # output_dir = "/home/ngsci/project/NEJM_benchmark/waveforms_med_beat/"
        # # List all files in the input directory
        # ecg_files = os.listdir(output_dir)
        # self.label_file = self.label_file[self.label_file["ecg_id"].isin(ecg_files)].reset_index()
        
        self.train_dataset = self.label_file[self.label_file["split"]=="train"].reset_index()
        self.val_dataset = self.label_file[self.label_file["split"]=="valid"].reset_index()
        self.test_dataset = self.label_file[self.label_file["split"]=="test"].reset_index()
        
        if self.class_balance:
            # This data is highly imbalanced!
            # Introduce a method to deal with class imbalance (hint: think about your data loader)
            if self.class_balance:
                y_train_sampler = np.array(self.train_dataset['y'])
            
            y_train_sampler_unique = np.unique(y_train_sampler)

            class_sample_count = np.array([len(np.where(y_train_sampler==t)[0]) for t in y_train_sampler_unique])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[list(y_train_sampler_unique).index(t)] for t in y_train_sampler])

            samples_weight = torch.from_numpy(samples_weight)
            self.train_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        
        # TODO Consider enabling 1D transforms
        self.train = OneDim_Dataset(dataset=self.train_dataset, transforms = None, covars=self.covars, scaler=self.covars_scaler)
        self.val = OneDim_Dataset(dataset=self.val_dataset, transforms = None, covars=self.covars, scaler=self.covars_scaler)
        self.test = OneDim_Dataset(dataset=self.test_dataset, transforms = None, covars=self.covars, scaler=self.covars_scaler)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

class OneDim_Dataset(torch.utils.data.Dataset):
    """
        Pytorch Dataset for 1D ECG representation dataset. Loads ECGs and attaches label.
    """

    def __init__(self, dataset, transforms, img_size=[256, 256], num_images=1, as_rgb=False, covars=[], scaler=None): #normalize,
        self.dataset = dataset
        self.transform = transforms
        #self.normalize = normalize
        self.img_size = img_size
        self.num_images = num_images
        self.as_rgb = as_rgb
        self.covars = covars
        self.covars_scaler = scaler
             
        print(self.get_summary_statement())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        ecgid = self.dataset.iloc[idx]['ecg_id'].split(".")[0]
        #sample_path = f"/home/ngsci/project/NEJM_benchmark/waveforms_img/{ecgid}.png"
        sample_path = f"/home/ngsci/project/NEJM_benchmark/waveforms/{ecgid}.npy"
        #sample_path = f"/home/ngsci/project/NEJM_benchmark/waveforms_med_beat/{ecgid}.npy"
        sample = {}
        #sample["x"] = Image.open(sample_path)
        sample["x"] = torch.Tensor(np.load(sample_path)).float()

        if self.as_rgb:
            sample["x"] = sample["x"].convert('RGB')

        if self.transform is not None:
            sample["x"] = self.transform(sample["x"])
        
        if len(self.covars)>0: 
            covars_data = self.dataset.iloc[idx][self.covars]
            # Normalize the covariates
            covars_data = self.covars_scaler.transform([covars_data])
            covars_data = covars_data.flatten()
            sample['cov'] = covars_data

        sample["y"] = self.dataset.iloc[idx]['y']
        sample['id'] = ecgid

        return sample

    def get_summary_statement(self):
        num_patients = len(self.dataset['ecg_id']) #len(set([d['ecgid'] for d in self.dataset]))
        num_outcome = sum(self.dataset['y']) #sum([d['y'] for d in self.dataset])
        return "ECG Dataset. {} ECGs ({} with outcome)".format(len(self.dataset),  num_outcome)


#####################################
###### Dataloader CPH200A ###########
#####################################

# class MedMNIST2D(MedMNIST):

#     def __getitem__(self, index):
#         '''
#         return: (without transform/target_transofrm)
#             img: PIL.Image
#             target: np.array of `L` (L=1 for single-label)
#         '''
#         img, target = self.imgs[index], self.labels[index].astype(int)
#         img = Image.fromarray(img)

#         if self.as_rgb:
#             img = img.convert('RGB')

#         if self.transform is not None:
#             img = self.transform(img)

#         if self.target_transform is not None:
#             target = self.target_transform(target)

#         return img, target   

# if self.class_balance or self.class_balance_risk:
#             # This data is highly imbalanced!
#             # Introduce a method to deal with class imbalance (hint: think about your data loader)
#             if self.class_balance:
#                 y_train_sampler = np.array([train_sample["y_seq"][0] for train_sample in self.train])
#             elif self.class_balance_risk:
#                 y_train_sampler = np.array([train_sample["y_seq"][5] for train_sample in self.train])
#             y_train_sampler_unique = np.unique(y_train_sampler)

#             class_sample_count = np.array([len(np.where(y_train_sampler==t)[0]) for t in y_train_sampler_unique])
#             weight = 1. / class_sample_count
#             samples_weight = np.array([weight[list(y_train_sampler_unique).index(t)] for t in y_train_sampler])

#             samples_weight = torch.from_numpy(samples_weight)
#             self.train_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

class NLST_Dataset(torch.utils.data.Dataset):
    """
        Pytorch Dataset for NLST dataset. Loads preprocesses data from disk and applies data augmentation. Generates masks from bounding boxes stored in metadata..
    """

    def __init__(self, dataset, transforms, normalize, img_size=[256, 256], num_images=200):
        self.dataset = dataset
        self.transform = transforms
        self.normalize = normalize
        self.img_size = img_size
        self.num_images = num_images

        print(self.get_summary_statement())

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample_path = self.dataset[idx]['path']
        sample = torch.load(sample_path)
        orig_pixel_spacing = torch.diag(torch.tensor(sample['pixel_spacing'] + [1]))
        num_slices = sample['x'].size()[0]

        right_side_cancer = sample['cancer_laterality'][0] == 1 and sample['cancer_laterality'][1] == 0
        left_side_cancer = sample['cancer_laterality'][1] == 1 and sample['cancer_laterality'][0] == 0

        # TODO: You can modify the data loading of the bounding boxes to suit your localization method.
        # Hint: You may want to use the "cancer_laterality" field to localize the cancer coarsely.

        if not sample['has_localization']:
            sample['bounding_boxes'] = None

        mask = self.get_scaled_annotation_mask(sample['bounding_boxes'], CACHE_IMG_SIZE + [num_slices])

        subject = tio.Subject( {
            'x': tio.ScalarImage(tensor=sample['x'].unsqueeze(0).to(torch.double), affine=orig_pixel_spacing),
            'mask': tio.LabelMap(tensor=mask.to(torch.double), affine=orig_pixel_spacing)
        })

        '''
            TorchIO will consistently apply the data augmentations to the image and mask, so that they are aligned. Note, the 'bounding_boxes' item will be wrong after after random transforms (e.g. rotations) in this implementation. 
        '''
        try:
            subject = self.transform(subject)
        except:
            raise Exception("Error with subject {}".format(sample_path))

        sample['x'], sample['mask'] = subject['x']['data'].to(torch.float), subject['mask']['data'].to(torch.float)
        ## Normalize volume to have 0 pixel mean and unit variance
        sample['x'] = self.normalize(sample['x'])

        sample['lung_rads'] = self.dataset[idx]['lung_rads']
        ## Remove potentially none items for batch collation
        del sample['bounding_boxes']

        return sample

    def get_summary_statement(self):
        num_patients = len(set([d['pid'] for d in self.dataset]))
        num_cancer = sum([d['y'] for d in self.dataset])
        num_cancer_year_1 = sum([d['y_seq'][0] for d in self.dataset])
        return "NLST Dataset. {} exams ({} with cancer in one year, {} cancer ever) from {} patients".format(len(self.dataset), num_cancer_year_1, num_cancer, num_patients)


class PathMnist(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for PathMnist dataset. This will download the dataset, prepare data loaders and apply
        data augmentation.
    """
    def __init__(self, use_data_augmentation=False, batch_size=32, num_workers=8, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.use_data_augmentation = use_data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.prepare_data_transforms()

    def prepare_data_transforms(self):
        '''
            Prepare data transforms for train and test data.
            Note, you may want to apply data augmentation (see torchvision) for the train data.
        '''
        self.test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        if self.use_data_augmentation:
            # TODO: Implement some data augmentatons
            self.train_transform = torchvision.transforms.Compose([
                            torchvision.transforms.RandomHorizontalFlip(),
                            torchvision.transforms.RandomVerticalFlip(),
                            torchvision.transforms.RandomRotation(10),  
                            # torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  
                            # torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), 
                            torchvision.transforms.ToTensor(),  
                            # torchvision.transforms.Normalize(mean=[0.5], std=[0.5])  
                        ])
        else:
            self.train_transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ])

    def prepare_data(self):
        medmnist.PathMNIST(root='../data', split='train', download=True, transform=self.train_transform)
        medmnist.PathMNIST(root='../data', split='val', download=True, transform=self.test_transform)
        medmnist.PathMNIST(root='../data', split='test', download=True, transform=self.test_transform)

    def setup(self, stage=None):
        self.train = medmnist.PathMNIST(root='../data', split='train', download=True, transform=self.train_transform)
        self.val = medmnist.PathMNIST(root='../data', split='val', download=True, transform=self.test_transform)
        self.test = medmnist.PathMNIST(root='../data', split='test', download=True, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

# Voxel spacing is space between pixels in orig 512x512xN volumes
# "pixel_spacing" stored in sample dicts is also in orig 512x512xN volumes
VOXEL_SPACING = (0.703125, 0.703125, 2.5)
CACHE_IMG_SIZE = [256, 256]

class NLST(pl.LightningDataModule):
    """
        Pytorch Lightning DataModule for NLST dataset. This will load the dataset, as used in https://ascopubs.org/doi/full/10.1200/JCO.22.01345.

        The dataset has been preprocessed for you fit on each CPH-App nodes NVMe SSD drives for faster experiments.
    """



    def __init__(
            self,
            use_data_augmentation=False,
            batch_size=6,
            num_workers=8,
            nlst_metadata_path="/wynton/protected/project/cph/cornerstone/nlst-metadata/full_nlst_google.json",
            valid_exam_path="/wynton/protected/project/cph/cornerstone/nlst-metadata/valid_exams.p",
            nlst_dir="/scratch/datasets/nlst/preprocessed",
            lungrads_path="/wynton/protected/project/cph/cornerstone/nlst-metadata/nlst_acc2lungrads.p",
            num_images=200,
            max_followup=6,
            img_size = [256, 256],
            class_balance=False,
            class_balance_risk=False,
            **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.use_data_augmentation = use_data_augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_followup = max_followup

        self.nlst_metadata_path = nlst_metadata_path
        self.nlst_dir = nlst_dir
        self.num_images = num_images
        self.img_size = img_size
        self.valid_exam_path = valid_exam_path
        self.class_balance = class_balance
        self.class_balance_risk = class_balance_risk
        self.lungrads_path = lungrads_path

        self.prepare_data_transforms()

    def prepare_data_transforms(self):
        '''
            Prepare data transforms for train and test data.
            Note, you may want to apply data augmentation (see torchvision) for the train data.
        '''
        resample = tio.transforms.Resample(target=VOXEL_SPACING)
        padding = tio.transforms.CropOrPad(
            target_shape=tuple(CACHE_IMG_SIZE + [self.num_images]), padding_mode=0
        )
        resize = tio.transforms.Resize(self.img_size + [self.num_images])


        self.train_transform = tio.transforms.Compose([
            resample,
            padding,
            resize
        ])

        if self.use_data_augmentation:
            # TODO: Support some data augmentations. Hint: consider using torchio.
            augment = tio.Compose([
                # tio.RandomAnisotropy(p=0.25),              # make images look anisotropic 25% of times
                tio.RandomAffine(degrees=10),
                tio.RandomFlip(),
                # tio.RandomNoise(p=0.25),                   # Gaussian noise 25% of times
                # tio.RandomGamma(p=0.5),                    # Randomly change contrast of an image by raising its values to the power gamma.
            ])

            self.train_transform = tio.transforms.Compose([
                resample,
                padding,
                resize,
                augment
            ])

        self.test_transform = tio.transforms.Compose([
            resample,
            padding,
            resize
        ])

        self.normalize = torchvision.transforms.Normalize(mean=[128.1722], std=[87.1849])

    def setup(self, stage=None):
        self.metadata = json.load(open(self.nlst_metadata_path, "r"))
        self.acc2lungrads = pickle.load(open(self.lungrads_path, "rb"))
        self.valid_exams = set(torch.load(self.valid_exam_path))
        self.train, self.val, self.test = [], [], []

        for mrn_row in tqdm.tqdm(self.metadata, position=0):
            pid, split, exams, pt_metadata = (
                mrn_row["pid"],
                mrn_row["split"],
                mrn_row["accessions"],
                mrn_row["pt_metadata"],
            )

            dataset = {"train": self.train, "dev": self.val, "test": self.test}[split]

            for exam_dict in exams:
                for series_id, series_dict in exam_dict["image_series"].items():

                    exam_str = "{}_{}".format(exam_dict["exam"], series_id)

                    if exam_str not in self.valid_exams:
                        continue


                    exam_int = int(
                        "{}{}{}".format(int(pid), int(exam_dict["screen_timepoint"]), int(series_id.split(".")[-1][-3:]))
                    )

                    y, y_seq, y_mask, time_at_event = self.get_label(pt_metadata, exam_dict["screen_timepoint"])
                    sample = {
                        "pid": pid,
                        "exam_str": exam_str,
                        "exam_int": exam_int,
                        "path": os.path.join(self.nlst_dir, exam_str + ".pt"),
                        "y": y,
                        "y_seq": y_seq,
                        "y_mask": y_mask,
                        "time_at_event": time_at_event,
                        # lung_rads 0 indicates LungRads 1 and 2 (negative), 1 indicates LungRads 3 and 4 (positive)
                        # Follows "Pinsky PF, Gierada DS, Black W, et al: Performance of lung-RADS in the National Lung Screening Trial: A retrospective assessment. Ann Intern Med 162: 485-491, 2015"
                        "lung_rads": self.acc2lungrads[exam_int],
                    }

                    dataset.append(sample)

        if self.class_balance or self.class_balance_risk:
            # This data is highly imbalanced!
            # Introduce a method to deal with class imbalance (hint: think about your data loader)
            if self.class_balance:
                y_train_sampler = np.array([train_sample["y_seq"][0] for train_sample in self.train])
            elif self.class_balance_risk:
                y_train_sampler = np.array([train_sample["y_seq"][5] for train_sample in self.train])
            y_train_sampler_unique = np.unique(y_train_sampler)

            class_sample_count = np.array([len(np.where(y_train_sampler==t)[0]) for t in y_train_sampler_unique])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[list(y_train_sampler_unique).index(t)] for t in y_train_sampler])

            samples_weight = torch.from_numpy(samples_weight)
            self.train_sampler = torch.utils.data.sampler.WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))


        self.train = NLST_Dataset(self.train, self.train_transform, self.normalize, self.img_size, self.num_images)
        self.val = NLST_Dataset(self.val, self.test_transform, self.normalize, self.img_size, self.num_images)
        self.test = NLST_Dataset(self.test, self.test_transform, self.normalize, self.img_size, self.num_images)

    def get_label(self, pt_metadata, screen_timepoint):
        days_since_rand = pt_metadata["scr_days{}".format(screen_timepoint)][0]
        days_to_cancer_since_rand = pt_metadata["candx_days"][0]
        days_to_cancer = days_to_cancer_since_rand - days_since_rand
        years_to_cancer = (
            int(days_to_cancer // 365) if days_to_cancer_since_rand > -1 else 100
        )
        days_to_last_followup = int(pt_metadata["fup_days"][0] - days_since_rand)
        years_to_last_followup = days_to_last_followup // 365
        y = years_to_cancer < self.max_followup
        y_seq = np.zeros(self.max_followup)
        cancer_timepoint = pt_metadata["cancyr"][0]
        if y:
            if years_to_cancer > -1:
                assert screen_timepoint <= cancer_timepoint
            time_at_event = years_to_cancer
            y_seq[years_to_cancer:] = 1
        else:
            time_at_event = min(years_to_last_followup, self.max_followup - 1)
        y_mask = np.array(
            [1] * (time_at_event + 1) + [0] * (self.max_followup - (time_at_event + 1))
        )
        assert len(y_mask) == self.max_followup
        return y, y_seq.astype("float64"), y_mask.astype("float64"), time_at_event

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=self.batch_size, num_workers=self.num_workers, sampler=self.train_sampler)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)


class NLST_Dataset(torch.utils.data.Dataset):
    """
        Pytorch Dataset for NLST dataset. Loads preprocesses data from disk and applies data augmentation. Generates masks from bounding boxes stored in metadata..
    """

    def __init__(self, dataset, transforms, normalize, img_size=[256, 256], num_images=200):
        self.dataset = dataset
        self.transform = transforms
        self.normalize = normalize
        self.img_size = img_size
        self.num_images = num_images

        print(self.get_summary_statement())

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample_path = self.dataset[idx]['path']
        sample = torch.load(sample_path)
        orig_pixel_spacing = torch.diag(torch.tensor(sample['pixel_spacing'] + [1]))
        num_slices = sample['x'].size()[0]

        right_side_cancer = sample['cancer_laterality'][0] == 1 and sample['cancer_laterality'][1] == 0
        left_side_cancer = sample['cancer_laterality'][1] == 1 and sample['cancer_laterality'][0] == 0

        # TODO: You can modify the data loading of the bounding boxes to suit your localization method.
        # Hint: You may want to use the "cancer_laterality" field to localize the cancer coarsely.

        if not sample['has_localization']:
            sample['bounding_boxes'] = None

        mask = self.get_scaled_annotation_mask(sample['bounding_boxes'], CACHE_IMG_SIZE + [num_slices])

        subject = tio.Subject( {
            'x': tio.ScalarImage(tensor=sample['x'].unsqueeze(0).to(torch.double), affine=orig_pixel_spacing),
            'mask': tio.LabelMap(tensor=mask.to(torch.double), affine=orig_pixel_spacing)
        })

        '''
            TorchIO will consistently apply the data augmentations to the image and mask, so that they are aligned. Note, the 'bounding_boxes' item will be wrong after after random transforms (e.g. rotations) in this implementation. 
        '''
        try:
            subject = self.transform(subject)
        except:
            raise Exception("Error with subject {}".format(sample_path))

        sample['x'], sample['mask'] = subject['x']['data'].to(torch.float), subject['mask']['data'].to(torch.float)
        ## Normalize volume to have 0 pixel mean and unit variance
        sample['x'] = self.normalize(sample['x'])

        sample['lung_rads'] = self.dataset[idx]['lung_rads']
        ## Remove potentially none items for batch collation
        del sample['bounding_boxes']

        return sample

    def get_scaled_annotation_mask(self, bounding_boxes, img_size=[256, 256, 200]):
        """
        Construct bounding box masks for annotations.

        Args:
            - bounding_boxes: list of dicts { 'x', 'y', 'width', 'height' }, where bounding box coordinates are scaled [0,1].
            - img_size per slice
        Returns:
            - mask of same size as input image, filled in where bounding box was drawn. If bounding_boxes = None, return empty mask. Values correspond to how much of a pixel lies inside the bounding box, as a fraction of the bounding box's area
        """
        H, W, Z = img_size
        if bounding_boxes is None:
            return torch.zeros((1, Z, H, W))

        masks = []
        for slice in bounding_boxes:
            slice_annotations = slice["image_annotations"]
            slice_mask = np.zeros((H, W))

            if slice_annotations is None:
                masks.append(slice_mask)
                continue

            for annotation in slice_annotations:
                single_mask = np.zeros((H, W))
                x_left, y_top = annotation["x"] * W, annotation["y"] * H
                x_right, y_bottom = (
                    min( x_left + annotation["width"] * W, W-1),
                    min( y_top + annotation["height"] * H, H-1),
                )

                # pixels completely inside bounding box
                x_quant_left, y_quant_top = math.ceil(x_left), math.ceil(y_top)
                x_quant_right, y_quant_bottom = math.floor(x_right), math.floor(y_bottom)

                # excess area along edges
                dx_left = x_quant_left - x_left
                dx_right = x_right - x_quant_right
                dy_top = y_quant_top - y_top
                dy_bottom = y_bottom - y_quant_bottom

                # fill in corners first in case they are over-written later by greater true intersection
                # corners
                single_mask[math.floor(y_top), math.floor(x_left)] = dx_left * dy_top
                single_mask[math.floor(y_top), x_quant_right] = dx_right * dy_top
                single_mask[y_quant_bottom, math.floor(x_left)] = dx_left * dy_bottom
                single_mask[y_quant_bottom, x_quant_right] = dx_right * dy_bottom

                # edges
                single_mask[y_quant_top:y_quant_bottom, math.floor(x_left)] = dx_left
                single_mask[y_quant_top:y_quant_bottom, x_quant_right] = dx_right
                single_mask[math.floor(y_top), x_quant_left:x_quant_right] = dy_top
                single_mask[y_quant_bottom, x_quant_left:x_quant_right] = dy_bottom

                # completely inside
                single_mask[y_quant_top:y_quant_bottom, x_quant_left:x_quant_right] = 1

                # in case there are multiple boxes, add masks and divide by total later
                slice_mask += single_mask
                    
            masks.append(slice_mask)

        return torch.Tensor(np.array(masks)).unsqueeze(0)

    def get_summary_statement(self):
        num_patients = len(set([d['pid'] for d in self.dataset]))
        num_cancer = sum([d['y'] for d in self.dataset])
        num_cancer_year_1 = sum([d['y_seq'][0] for d in self.dataset])
        return "NLST Dataset. {} exams ({} with cancer in one year, {} cancer ever) from {} patients".format(len(self.dataset), num_cancer_year_1, num_cancer, num_patients)