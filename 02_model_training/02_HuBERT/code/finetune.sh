#!/bin/bash

# Commands to run to perform the fine-tuning of HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All dataset as done in the paper
# Standard train/val/test splits are used
# Altoigh previous commands actually reproduce the results we obtain on PTB-XL All, we consider this dataset only as an example
# More useful information available with:   python finetune.py --help

#############################################################################
#################################### ACS ####################################
#############################################################################

## Finetune complete model
python finetune.py 3 /home/ngsci/project/ACS_benchmark/data/train_ids_labels_with_covars_all_final_HuBERT_cath.csv /home/ngsci/project/ACS_benchmark/data/val_ids_labels_with_covars_all_final_HuBERT_cath.csv --vocab_size=2 --patience=8 --batch_size=64 --target_metric=auroc \
    --load_path=/home/ngsci/project/ACS_benchmark/HuBERT-ECG/hubert_ecg_large.pt \
    --training_steps=7000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=16 --classifier_hidden_size 1024 --model_dropout_mult=-2 --val_interval=50 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ACS_full
    
#############################################################################
################################ Template ###################################
#############################################################################

python finetune.py 3 /path/to/ptb_all_train.csv /path/to/ptb_all_val.csv 71 8 64 auroc \
    --load_path=path/to/pretrained/model/small/hubert_ecg_small.pt \
    --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=500 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=SMALL_ptbxl_all

python finetune.py 2 /path/to/ptb_all_train.csv /path/to/ptb_all_val.csv 71 8 64 auroc \
    --load_path=path/to/pretrained/model/base/hubert_ecg_base.pt \
    --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=12 --model_dropout_mult=-2 --val_interval=500 \
    --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=BASE_ptbxl_all

python finetune.py 3 /path/to/ptb_all_train.csv /path/to/ptb_all_val.csv 71 8 64 auroc \
    --load_path=path/to/pretrained/model/large/hubert_ecg_large.pt \
    --training_steps=70000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=16 --model_dropout_mult=-2 --val_interval=500 \
    --finetuning_layerdrop=0.1 --random_crop --wandb_run_name=LARGE_ptbxl_all

