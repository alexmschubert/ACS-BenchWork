#!/bin/bash

####################################################
## Evaluation of models
####################################################

# ACS
python test.py /home/ngsci/project/ACS_benchmark/data/val_ids_labels_with_covars_all_final_HuBERT_cath.csv /home/ngsci/project/ACS_benchmark/waveforms_12lead_10sec/ 64 \
    /home/ngsci/project/ACS_benchmark/02_model_training/02_HuBERT/finetuned_models/hubert_3_iteration_300_finetuned_simdmsnv_acs_full.pt \
    --downsampling_factor=5 \
    --label_start_index=4 \
    --tta \
    --save_id=acs_large_full \
    --tta_aggregation=max
    

####################################################
## Templates
####################################################

# Command to run to test HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All test set
# More useful information with:   python test.py --help
# Finetuned models are available

python test.py /path/to/ptb_all_test.csv . 64 \
    ../path/to/finetuned/model/hubert_small_12.5k_ptbAll.pt \
    --downsampling_factor=5 \
    --label_start_index=4 \
    --tta \
    --save_id=ptb_all_small \
    --tta_aggregation=max

python test.py /path/to/ptb_all_test.csv . 64 \
    ../path/to/finetuned/model/hubert_base_9k_ptbAll.pt \
    --downsampling_factor=5 \
    --label_start_index=4 \
    --tta \
    --save_id=ptb_all_base \
    --tta_aggregation=max

python test.py /path/to/ptb_all_test.csv . 64 \
    ../path/to/finetuned/model/hubert_large_8.5k_ptbAll.pt \
    --downsampling_factor=5 \
    --label_start_index=4 \
    --tta \
    --save_id=ptb_all_large \
    --tta_aggregation=max

