import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from multiprocessing import Pool
import os

import matplotlib.pyplot as plt

if __name__ == "__main__":
    waveform_array = np.load('/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-waveform.npy')
    ed_encounter = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/ed-encounter.csv')
    ecg_metadata = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-metadata.csv')
    ecg_npy = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-npy-index.csv')
    patient = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/patient.csv')
    ecg_to_ed_enc = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-ed-enc.csv')
    
    ###########################################
    ## Get encounter ID into the dataframe 
    ###########################################

    # Check if the columns in ecg_to_ed_enc (except 'ecg_id') are not in ecg_metadata
    columns_to_check = [col for col in ecg_to_ed_enc.columns if col != 'ecg_id']
    if not any(col in ecg_metadata.columns for col in columns_to_check):
        # Merge the DataFrames
        merged_df = pd.merge(ecg_metadata, ecg_to_ed_enc, on='ecg_id', how='left')
        print('done')
    else:
        print("Merge not performed: Columns from ecg_to_ed_enc already exist in ecg_metadata.")

    ###########################################
    ## Get encounter information into dataframe 
    ###########################################

    # Check if the columns in ecg_to_ed_enc (except 'ecg_id') are not in ecg_metadata
    ed_encounter_new = ed_encounter.drop('patient_ngsci_id', axis=1)
    columns_to_check = [col for col in ed_encounter.columns if col not in ['ed_enc_id','patient_ngsci_id']]
    if not any(col in merged_df.columns for col in columns_to_check):
        # Merge the DataFrames
        ecg_analysis_df = pd.merge(merged_df, ed_encounter_new, on='ed_enc_id', how='left')
        print('done')
    else:
        print("Merge not performed: Columns from merged_df already exist in ed_encounter.")
    
    print(ecg_analysis_df.columns)

    ###############################################################
    ## Restrict sample to patients included in QJE paper 
    ###############################################################
    print(ecg_analysis_df.shape)
    ecg_analysis_df_included = ecg_analysis_df[ecg_analysis_df['exclude']==False]
    print(ecg_analysis_df_included.shape)
    
    ###############################################################
    ## Restrict sample to patients that were tested
    ###############################################################
    print(ecg_analysis_df_included.shape)
    ecg_analysis_df_tested = ecg_analysis_df_included[ecg_analysis_df_included['test_010_day']==True]
    print(ecg_analysis_df_tested.shape)
    
    ###############################################################
    ## Evaluate performance of STE based AMI detection
    ###############################################################

    def evaluate_testing_performance(testing_label='has_st_eleva'):
        print()
        print(f'Evaluation for label: {testing_label}')
        print()
        ## Depict prevalence of label among included ECGs
        prevalence = round((ecg_analysis_df_included[ecg_analysis_df_included[testing_label]==True].shape[0]) / (ecg_analysis_df_included.shape[0]),4)*100
        print(f'Prevalence of {testing_label} among all included ECGs is {prevalence}%')

        ## Testing among those with label
        label_population = ecg_analysis_df_included[ecg_analysis_df_included[testing_label]==True]
        testing_rate = round(label_population['test_010_day'].sum()/(label_population.shape[0]),4)*100
        print(f'The testing rate among those with {testing_label} is {testing_rate}%')

        ## Depict prevalence of label among tested ECGs
        prevalence = round((ecg_analysis_df_tested[ecg_analysis_df_tested[testing_label]==True].shape[0]) / (ecg_analysis_df_tested.shape[0]),4)*100
        print(f'Prevalence of {testing_label} among all ECGs corresponding to visits with subsequent definitive AMI testing is {prevalence}%')

        # Testing yield if patient with ECG with label got tested
        label_population_tpr = ecg_analysis_df_tested[ecg_analysis_df_tested[testing_label]==True]
        testing_tpr = round((label_population_tpr['stent_or_cabg_010_day'].sum())/(label_population_tpr.shape[0])*100,2)
        print(f'The testing yield among those with {testing_label} is {testing_tpr}%')

        # Share of blockages detected
        label_population_tpr = ecg_analysis_df_tested[ecg_analysis_df_tested[testing_label]==True]
        testing_tpr = round((label_population_tpr['stent_or_cabg_010_day'].sum())/(ecg_analysis_df_tested['stent_or_cabg_010_day'].sum())*100,2)
        print(f'Of all identified ACS among those with {testing_tpr}% presented with {testing_label} in their ECG')

        # Testing yield if patient with ECG without label got tested
        no_label_population_tpr = ecg_analysis_df_tested[ecg_analysis_df_tested[testing_label]==False]
        testing_tpr = round((no_label_population_tpr['stent_or_cabg_010_day'].sum())/(no_label_population_tpr.shape[0])*100,2)
        print(f'The testing yield among those without {testing_label} is {testing_tpr}%')
        print()


    ecg_analysis_df_included['ste_std_twi'] = ecg_analysis_df_included[['has_st_eleva', 'has_twave_inver', 'has_depress']].any(axis=1)
    ecg_analysis_df_tested['ste_std_twi'] = ecg_analysis_df_tested[['has_st_eleva', 'has_twave_inver', 'has_depress']].any(axis=1)

    labels = ['has_st_eleva', 'has_twave_inver', 'has_depress', 'ste_std_twi']

    for label in labels:
        evaluate_testing_performance(label)    
    
    ### Create train / val / test split for tested population, split by patient 

    patient_ids = ecg_analysis_df_tested.patient_ngsci_id.unique() 
    print(len(patient_ids))
    #print(ecg_analysis_df_tested.head())
    print(patient_ids)

    # Splitting the ids into train (50%) and temp (50%)
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.5, random_state=0)

    # Splitting the temp into val (20% of total) and test (30% of total)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.6, random_state=0)

    # Creating 3 versions of the dataframe
    df_train = ecg_analysis_df_tested[ecg_analysis_df_tested['patient_ngsci_id'].isin(train_ids)][['ecg_id','stent_or_cabg_010_day']]
    df_train['split'] = 'train'
    df_train['ecg_id'] = df_train['ecg_id'].astype(str).apply(lambda x: x + '.npy')
    df_val = ecg_analysis_df_tested[ecg_analysis_df_tested['patient_ngsci_id'].isin(val_ids)][['ecg_id','stent_or_cabg_010_day']]
    df_val['split'] = 'valid'
    df_val['ecg_id'] = df_val['ecg_id'].astype(str).apply(lambda x: x + '.npy')
    df_test = ecg_analysis_df_tested[ecg_analysis_df_tested['patient_ngsci_id'].isin(test_ids)][['ecg_id','stent_or_cabg_010_day']]
    df_test['split'] = 'test'
    df_test['ecg_id'] = df_test['ecg_id'].astype(str).apply(lambda x: x + '.npy')
    print(df_test.head())

    df_all = pd.concat([df_train, df_val, df_test])

    df_train.to_csv('train_ids_labels.csv')
    df_val.to_csv('val_ids_labels.csv')
    df_test.to_csv('test_ids_labels.csv')
    df_all.to_csv('all_ids_labels_tested.csv')


    # Extract respective ecg ids
    train_ids = df_train['ecg_id'].tolist()
    val_ids = df_val['ecg_id'].tolist()
    test_ids = df_test['ecg_id'].tolist()
    
    ecg_analysis_df_untested = ecg_analysis_df_included[ecg_analysis_df_included['test_010_day']==False]
    
    patient_ids = ecg_analysis_df_untested.patient_ngsci_id.unique() 
    print(len(patient_ids))
    #print(ecg_analysis_df_tested.head())
    print(patient_ids)

    # Splitting the ids into train (50%) and temp (50%)
    train_ids, temp_ids = train_test_split(patient_ids, test_size=0.5, random_state=0)

    # Splitting the temp into val (20% of total) and test (30% of total)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.6, random_state=0)

    # Creating 3 versions of the dataframe
    df_train = ecg_analysis_df_untested[ecg_analysis_df_untested['patient_ngsci_id'].isin(train_ids)][['ecg_id','macetrop_pos_or_death_030']]
    df_train['split'] = 'train'
    df_train['ecg_id'] = df_train['ecg_id'].astype(str).apply(lambda x: x + '.npy')
    df_val = ecg_analysis_df_untested[ecg_analysis_df_untested['patient_ngsci_id'].isin(val_ids)][['ecg_id','macetrop_pos_or_death_030']]
    df_val['split'] = 'valid'
    df_val['ecg_id'] = df_val['ecg_id'].astype(str).apply(lambda x: x + '.npy')
    df_test = ecg_analysis_df_untested[ecg_analysis_df_untested['patient_ngsci_id'].isin(test_ids)][['ecg_id','macetrop_pos_or_death_030']]
    df_test['split'] = 'test'
    df_test['ecg_id'] = df_test['ecg_id'].astype(str).apply(lambda x: x + '.npy')
    print(df_test.head())

    df_all = pd.concat([df_train, df_val, df_test])

    df_train.to_csv('train_ids_labels_untested.csv')
    df_val.to_csv('val_ids_labels_untested.csv')
    df_test.to_csv('test_ids_labels_untested.csv')
    df_all.to_csv('all_ids_labels_untested.csv')
    
    # Function to process each ECG waveform
    def process_ecg(index):
        ecg = waveform_array[index].copy()

        # Set mode to 0
        for i in range(ecg.shape[0]):
            sequence = ecg[i]
            sequence_mode = mode(sequence)[0][0]
            ecg[i] -= sequence_mode

        info = ecg_npy[ecg_npy['npy_index'] == index]
        ecg_id = info['ecg_id'].item()

        # Define sampling rate and time vector
        sampling_rate = 100  # 100 Hz
        time = np.linspace(0, 10, 1000)  # 10 seconds

        # Plot each ECG lead separately
        fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        leads = ['V1', 'II', 'V5']

        # Set a common y-axis range
        y_min = -2
        y_max = 2.5

        for i in range(3):
            axs[i].plot(time, ecg[i], label=leads[i])
            axs[i].set_title(leads[i])
            axs[i].set_ylabel('Amplitude')
            axs[i].set_ylim(y_min, y_max)
            axs[i].grid(which='both', linestyle='-', linewidth=0.5)
            axs[i].legend(loc='upper right')

        axs[-1].set_xlabel('Time (s)')

        plt.tight_layout()
        plt.savefig(f"/home/ngsci/project/NEJM_benchmark/waveforms_img/{ecg_id}.png")
        plt.close(fig)  # Close the plot to free memory

    # Assuming waveform_array and ecg_npy are defined
    num_processes = os.cpu_count() - 1 # Number of available CPU cores
    print(num_processes)


    # Create a Pool of workers

    with Pool(num_processes) as pool:
        list(tqdm(pool.imap(process_ecg, range(waveform_array.shape[0])), total=waveform_array.shape[0]))

    # imap is used for lazy iteration over the data
