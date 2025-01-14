#!/bin/bash

# Commands to run to perform the fine-tuning of HuBERT-ECG SMALL, BASE and LARGE models on the PTB-XL All dataset as done in the paper
# Standard train/val/test splits are used
# Altoigh previous commands actually reproduce the results we obtain on PTB-XL All, we consider this dataset only as an example
# More useful information available with:   python finetune.py --help

#############################################################################
################################# Tamil Nadu ################################
#############################################################################

## Only update classification layer
python finetune.py 3 /home/ngsci/project/Tamil-Nadu/Structured_data/20241204_tn_train_set_HuBERT.csv /home/ngsci/project/Tamil-Nadu/Structured_data/20241204_tn_val_set_HuBERT.csv --vocab_size=2 --patience=8 --batch_size=64 --target_metric=auroc \
    --load_path=/home/ngsci/project/NEJM_benchmark/HuBERT-ECG/hubert_ecg_large.pt --ecg_dir_path_train /home/ngsci/project/Tamil-Nadu/12-lead-ecgs --ecg_dir_path_val /home/ngsci/project/Tamil-Nadu/12-lead-ecgs\
    --training_steps=5000 --downsampling_factor=10 --label_start_index=4 \
    --transformer_blocks_to_unfreeze=0 --classifier_hidden_size 1024 --model_dropout_mult=-2 --val_interval=50 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_TN_RWMA_last_layer

#############################################################################
#################################### ACS ####################################
#############################################################################

## Only update classification layer
python finetune.py 3 /home/ngsci/project/NEJM_benchmark/train_ids_labels_with_covars_all_final_HuBERT_cath.csv /home/ngsci/project/NEJM_benchmark/val_ids_labels_with_covars_all_final_HuBERT_cath.csv --vocab_size=2 --patience=8 --batch_size=64 --target_metric=auroc \
    --load_path=/home/ngsci/project/NEJM_benchmark/HuBERT-ECG/hubert_ecg_large.pt \
    --training_steps=7000 --downsampling_factor=5 --label_start_index=4 \
    --transformer_blocks_to_unfreeze=0 --classifier_hidden_size 1024 --model_dropout_mult=-2 --val_interval=50 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ACS_last_layer

## Finetune complete model
python finetune.py 3 /home/ngsci/project/NEJM_benchmark/train_ids_labels_with_covars_all_final_HuBERT_cath.csv /home/ngsci/project/NEJM_benchmark/val_ids_labels_with_covars_all_final_HuBERT_cath.csv --vocab_size=2 --patience=8 --batch_size=64 --target_metric=auroc \
    --load_path=/home/ngsci/project/NEJM_benchmark/HuBERT-ECG/hubert_ecg_large.pt \
    --training_steps=7000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=16 --classifier_hidden_size 1024 --model_dropout_mult=-2 --val_interval=50 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ACS_full

#############################################################################
################################### MACE ####################################
#############################################################################


## Only update classification layer
python finetune.py 3 /home/ngsci/project/NEJM_benchmark/train_ids_labels_untested_with_covars_all_final_HuBERT.csv /home/ngsci/project/NEJM_benchmark/val_ids_labels_untested_with_covars_all_final_HuBERT.csv --vocab_size=2 --patience=8 --batch_size=64 --target_metric=auroc \
    --load_path=/home/ngsci/project/NEJM_benchmark/HuBERT-ECG/hubert_ecg_large.pt \
    --training_steps=7000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=0 --classifier_hidden_size 1024 --model_dropout_mult=-2 --val_interval=200 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_MACE_last_layer

## Finetune complete model
python finetune.py 3 /home/ngsci/project/NEJM_benchmark/train_ids_labels_untested_with_covars_all_final_HuBERT.csv /home/ngsci/project/NEJM_benchmark/val_ids_labels_untested_with_covars_all_final_HuBERT.csv --vocab_size=2 --patience=8 --batch_size=64 --target_metric=auroc \
    --load_path=/home/ngsci/project/NEJM_benchmark/HuBERT-ECG/hubert_ecg_large.pt \
    --training_steps=7000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=16 --classifier_hidden_size 1024 --model_dropout_mult=-2 --val_interval=200 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_MACE_full


##############################################
## Only update classification layer

python finetune.py 3 /home/ngsci/project/NEJM_benchmark/train_ids_labels_with_covars_all_final_HuBERT_cath.csv /home/ngsci/project/NEJM_benchmark/val_ids_labels_with_covars_all_final_HuBERT_cath.csv 2 --patience=8 --batch_size=64 --target_metric=auroc \
    --load_path=/home/ngsci/project/NEJM_benchmark/HuBERT-ECG/hubert_ecg_large.pt \
    --training_steps=7000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=200 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ACS_all

python finetune.py 3 /home/ngsci/project/NEJM_benchmark/train_ids_labels_untested_with_covars_all_final_HuBERT.csv /home/ngsci/project/NEJM_benchmark/val_ids_labels_untested_with_covars_all_final_HuBERT.csv 2 --patience=8 --batch_size=64 --target_metric=auroc \
    --load_path=/home/ngsci/project/NEJM_benchmark/HuBERT-ECG/hubert_ecg_large.pt \
    --training_steps=7000 --downsampling_factor=5 --label_start_index=4 --use_loss_weights \
    --transformer_blocks_to_unfreeze=8 --model_dropout_mult=-2 --val_interval=200 \
    --finetuning_layerdrop=0.0 --random_crop --wandb_run_name=LARGE_ACS_all
    
#############################################################################
################################### Before ##################################
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

