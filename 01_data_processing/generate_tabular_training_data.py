import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy.stats import mode
from scipy import stats
from multiprocessing import Pool
import neurokit2 as nk
from biosppy.signals import ecg
import biosppy
import os

import matplotlib.pyplot as plt

from confidenceinterval import roc_auc_score, accuracy_score


if __name__ == "__main__":

    # ---------------------------------------------------------------------
    # 1. Load Data
    # ---------------------------------------------------------------------
    waveform_array = np.load('/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-waveform.npy')
    ed_encounter = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/ed-encounter.csv')
    ecg_metadata = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-metadata.csv')
    ecg_npy = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-npy-index.csv')
    patient = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/patient.csv')
    ecg_to_ed_enc = pd.read_csv('/home/ngsci/datasets/ed-bwh-ecg/v1/ecg-ed-enc.csv')

    # ---------------------------------------------------------------------
    # 2. Clean & Merge DataFrames
    # ---------------------------------------------------------------------
    ecg_metadata['ecg_id_new'] = ecg_metadata['ecg_id'].str[3:]
    ecg_npy['ecg_id_new'] = ecg_npy['ecg_id'].str[3:]
    ecg_to_ed_enc['ecg_id_new'] = ecg_to_ed_enc['ecg_id'].str[3:]

    ## Get encounter ID into the dataframe 
    # Check if the columns in ecg_to_ed_enc (except 'ecg_id') are not in ecg_metadata
    ecg_to_ed_enc_new = ecg_to_ed_enc.drop('ecg_id', axis=1)
    columns_to_check = [col for col in ecg_to_ed_enc.columns if col != 'ecg_id' and col != 'ecg_id_new']
    if not any(col in ecg_metadata.columns for col in columns_to_check):
        # Merge the DataFrames
        merged_df = pd.merge(ecg_metadata, ecg_to_ed_enc_new, on='ecg_id_new', how='left')
        print('done')
    else:
        print("Merge not performed: Columns from ecg_to_ed_enc already exist in ecg_metadata.")

    ## Get encounter information into dataframe 
    # Check if the columns in ecg_to_ed_enc (except 'ecg_id') are not in ecg_metadata
    ed_encounter_new = ed_encounter.drop('patient_ngsci_id', axis=1)
    columns_to_check = [col for col in ed_encounter.columns if col not in ['ed_enc_id','patient_ngsci_id']]
    if not any(col in merged_df.columns for col in columns_to_check):
        # Merge the DataFrames
        ecg_analysis_df = pd.merge(merged_df, ed_encounter_new, on='ed_enc_id', how='left')
        print('done')
    else:
        print("Merge not performed: Columns from merged_df already exist in ed_encounter.")
        

    ## Get patient info into dataframe 
    # Check if the columns in ecg_to_ed_enc (except 'ecg_id') are not in ecg_metadata
    columns_to_check = [col for col in patient.columns if col not in ['ed_enc_id','patient_ngsci_id']]
    if not any(col in merged_df.columns for col in columns_to_check):
        # Merge the DataFrames
        ecg_analysis_df = pd.merge(ecg_analysis_df, patient, on='patient_ngsci_id', how='left')
        print('done')
    else:
        print("Merge not performed: Columns from merged_df already exist in ed_encounter.")
    
    # ---------------------------------------------------------------------
    # 3. Filter DataFrames into Tested / Untested (and apply any exclusions)
    # ---------------------------------------------------------------------

    # Exclude frail and sick patients who may not get treated for other reasons beyond the catheterization result    
    ecg_analysis_df_included_all = ecg_analysis_df[ecg_analysis_df['exclude_modeling']==False]
    ecg_analysis_df_included_tested_all = ecg_analysis_df_included_all[ecg_analysis_df_included_all['cath_010_day']==True]
    ecg_analysis_df_included_untested_all = ecg_analysis_df_included_all[ecg_analysis_df_included_all['cath_010_day']==False]

    # ---------------------------------------------------------------------
    # 4. Create Train/Val Splits: TESTED GROUP
    # ---------------------------------------------------------------------

    # get tested ECG files
    ecg_analysis_df_included_tested_all['female'] = (ecg_analysis_df_included_tested_all['sex'] == "Female").astype(int)
    patient_ids_tested = ecg_analysis_df_included_tested_all.patient_ngsci_id.unique() 

    # Split: 50% train, 50% val (example ratio)
    train_ids, val_ids = train_test_split(patient_ids_tested, test_size=0.3, random_state=0)

    # Create final DataFrames
    df_train_f = ecg_analysis_df_included_tested_all[ecg_analysis_df_included_tested_all["patient_ngsci_id"].isin(train_ids)].copy()
    df_train_f["split"] = "train"
    df_train_f["ecg_id_new"] = df_train_f["ecg_id_new"].astype(str) + ".npy"

    df_val_f = ecg_analysis_df_included_tested_all[ecg_analysis_df_included_tested_all["patient_ngsci_id"].isin(val_ids)].copy()
    df_val_f["split"] = "valid"
    df_val_f["ecg_id_new"] = df_val_f["ecg_id_new"].astype(str) + ".npy"

    # Combined (train + val)
    df_all_f = pd.concat([df_train_f, df_val_f], ignore_index=True)

    # Save CSVs
    df_train_f.to_csv("data/train_ids_labels_with_covars_all_final_cath.csv", index=False)
    df_val_f.to_csv("data/val_ids_labels_with_covars_all_final_cath.csv", index=False)
    df_all_f.to_csv("data/all_ids_labels_tested_with_covars_all_final_cath.csv", index=False)

    # ---------------------------------------------------------------------
    # 5. Format for HuBERT: TESTED GROUP
    # ---------------------------------------------------------------------
    # For HuBERT, combine train + val into "train_f_bert", keep "val_f_bert" as separate
    df_train_f_bert = df_train_f.copy()
    df_val_f_bert = df_val_f.copy()

    # Create a .npy filename
    df_train_f_bert["filename"] = df_train_f_bert["ecg_id"].str.replace("^ecg", "", regex=True) + ".npy"
    df_val_f_bert["filename"] = df_val_f_bert["ecg_id"].str.replace("^ecg", "", regex=True) + ".npy"

    df_train_f_bert = df_train_f_bert[["filename", "patient_ngsci_id", "ecg_id", "stent_or_cabg_010_day"]]
    df_val_f_bert = df_val_f_bert[["filename", "patient_ngsci_id", "ecg_id", "stent_or_cabg_010_day"]]

    # Create a complementary label (example)
    df_train_f_bert["no_stent_or_cabg_010_day"] = (~(df_train_f_bert["stent_or_cabg_010_day"] == 1)).astype(int)
    df_val_f_bert["no_stent_or_cabg_010_day"] = (~(df_val_f_bert["stent_or_cabg_010_day"] == 1)).astype(int)

    # Save CSVs
    df_train_f_bert.to_csv("data/train_ids_labels_with_covars_all_final_HuBERT_cath.csv", index=False)
    df_val_f_bert.to_csv("data/val_ids_labels_with_covars_all_final_HuBERT_cath.csv", index=False)

    # ---------------------------------------------------------------------
    # 6. Create Train/Val Splits: UNTESTED GROUP
    # ---------------------------------------------------------------------
    
    # Develop training files - for the untested group
    ecg_analysis_df_included_untested_all['female'] = (ecg_analysis_df_included_untested_all['sex'] == "Female").astype(int)
    patient_ids_untested = ecg_analysis_df_included_untested_all.patient_ngsci_id.unique() 

    # Split: 50% train, 50% val (example ratio)
    train_ids, val_ids = train_test_split(patient_ids_untested, test_size=0.3, random_state=0)

    # Create final DataFrames
    df_train_f = ecg_analysis_df_included_untested_all[ecg_analysis_df_included_untested_all["patient_ngsci_id"].isin(train_ids)].copy()
    df_train_f["split"] = "train"
    df_train_f["ecg_id"] = df_train_f["ecg_id"].astype(str) + ".npy"

    df_val_f = ecg_analysis_df_included_untested_all[ecg_analysis_df_included_untested_all["patient_ngsci_id"].isin(val_ids)].copy()
    df_val_f["split"] = "val"
    df_val_f["ecg_id"] = df_val_f["ecg_id"].astype(str) + ".npy"

    # Combined (train + val)
    df_all_f = pd.concat([df_train_f, df_val_f], ignore_index=True)

    # Save CSVs
    df_train_f.to_csv("data/train_ids_labels_untested_with_covars_all_final.csv", index=False)
    df_val_f.to_csv("data/val_ids_labels_untested_with_covars_all_final.csv", index=False)
    df_all_f.to_csv("data/all_ids_labels_untested_with_covars_all_final.csv", index=False)

    # ---------------------------------------------------------------------
    # 7. Format for HuBERT: UNTESTED GROUP
    # ---------------------------------------------------------------------
    df_train_f_bert = df_train_f.copy()
    df_val_f_bert = df_val_f.copy()

    # Create a .npy filename
    df_train_f_bert["filename"] = df_train_f_bert["ecg_id"].str.replace("^ecg", "", regex=True)
    df_val_f_bert["filename"] = df_val_f_bert["ecg_id"].str.replace("^ecg", "", regex=True)

    df_train_f_bert = df_train_f_bert[["filename", "patient_ngsci_id", "ecg_id", "macetrop_pos_or_death_030"]]
    df_val_f_bert = df_val_f_bert[["filename", "patient_ngsci_id", "ecg_id", "macetrop_pos_or_death_030"]]

    # Create a complementary label
    df_train_f_bert["no_macetrop_pos_or_death_030"] = (~(df_train_f_bert["macetrop_pos_or_death_030"] == 1)).astype(int)
    df_val_f_bert["no_macetrop_pos_or_death_030"] = (~(df_val_f_bert["macetrop_pos_or_death_030"] == 1)).astype(int)

    # Save CSVs
    df_train_f_bert.to_csv("data/train_ids_labels_untested_with_covars_all_final_HuBERT.csv", index=False)
    df_val_f_bert.to_csv("data/val_ids_labels_untested_with_covars_all_final_HuBERT.csv", index=False)