# Training commands

## Model pretraining

We pretrain the model to predict major adverse cardiovascular events (MACE) in the population without catheterization outcome.

```bash
python main.py --project_name benchmark_mace_state_v0 --train --trainer.max_epochs 50 --model_name "state" --dataset_name "ecg_seq" --use_data_augmentation False --state.init_lr 1e-4 --state.d_input 3 --state.d_output 1 --label_key "macetrop_pos_or_death_030" --label_file "/home/ngsci/project/ACS_benchmark/all_ids_labels_untested_with_covars_all_final.csv" --batch_size 32

python main.py --project_name benchmark_mace_resnet18_1d_final_vf --train --trainer.max_epochs 50 --model_name "resnet18_1d" --dataset_name "ecg_seq" --use_data_augmentation False --resnet18_1d.init_lr 1e-4 --label_key "macetrop_pos_or_death_030" --label_file "/home/ngsci/project/ACS_benchmark/all_ids_labels_untested_with_covars_all_final.csv"
```

Models reported in supplement:

```bash
python main.py --project_name benchmark_mace_resnet6_final_vf --train --trainer.max_epochs 50 --model_name "resnet6" --dataset_name "ecg_seq" --use_data_augmentation False --resnet6.init_lr 1e-4 --label_key "macetrop_pos_or_death_030" --label_file "/home/ngsci/project/ACS_benchmark/all_ids_labels_untested_with_covars_all_final.csv" 

python main.py --project_name benchmark_mace_resnet50_1d_final_vf --train --trainer.max_epochs 50 --model_name "resnet50_1d" --dataset_name "ecg_seq" --use_data_augmentation False --resnet50_1d.init_lr 1e-4 --label_key "macetrop_pos_or_death_030" --label_file "/home/ngsci/project/ACS_benchmark/all_ids_labels_untested_with_covars_all_final.csv"  
```


## ACS Model training

We train the final ACS prediction models, initializing on the weights of the MACE models.

```bash
python main.py --project_name benchmark_acs_state_v0 --train --trainer.max_epochs 50 --model_name "state" --dataset_name "ecg_seq" --use_data_augmentation False --state.init_lr 1e-4 --state.d_input 3 --state.d_output 1 --label_key "stent_or_cabg_010_day" --label_file "/home/ngsci/project/ACS_benchmark/all_ids_labels_tested_with_covars_all_final_cath.csv" --batch_size 32 --checkpoint_path "/home/ngsci/project/ACS_benchmark/ECG_img_benchmark/benchmark_mace_state_v0/0j51g744/checkpoints/epoch=36-step=14023.ckpt"

python main.py --project_name benchmark_acs_resnet18_1d_final_vf --train --trainer.max_epochs 50 --model_name "resnet18_1d" --dataset_name "ecg_seq" --use_data_augmentation False --resnet18_1d.init_lr 1e-4 --label_key "stent_or_cabg_010_day" --label_file "/home/ngsci/project/ACS_benchmark/all_ids_labels_tested_with_covars_all_final_cath.csv" --checkpoint_path "/home/ngsci/project/ACS_benchmark/ECG_img_benchmark/benchmark_mace_resnet18_1d_final_vf/s9niyfha/checkpoints/epoch=25-step=19682.ckpt" 
```

Models reported in supplement:

```bash
python main.py --project_name benchmark_acs_resnet6_final_vf --train --trainer.max_epochs 50 --model_name "resnet6" --dataset_name "ecg_seq" --use_data_augmentation False --resnet6.init_lr 1e-4 --label_key "stent_or_cabg_010_day" --label_file "/home/ngsci/project/ACS_benchmark/all_ids_labels_tested_with_covars_all_final_cath.csv" --checkpoint_path "/home/ngsci/project/ACS_benchmark/ECG_img_benchmark/benchmark_mace_ecgomi_final_vf/y3mxvcwt/checkpoints/epoch=9-step=7570.ckpt"

python main.py --project_name benchmark_acs_resnet18_1d_final_vf --train --trainer.max_epochs 50 --model_name "resnet18_1d" --dataset_name "ecg_seq" --use_data_augmentation False --resnet18_1d.init_lr 1e-4 --label_key "stent_or_cabg_010_day" --label_file "/home/ngsci/project/ACS_benchmark/all_ids_labels_tested_with_covars_all_final_cath.csv" --checkpoint_path "/home/ngsci/project/ACS_benchmark/ECG_img_benchmark/benchmark_mace_resnet18_1d_final_vf/s9niyfha/checkpoints/epoch=25-step=19682.ckpt" 
```

## Obtaining ACS predictions

Prediction using the models mentioned in the main text:

```bash
python prediction.py --project_name ACS_benchmark --experiment_name state_tested_v0 --trainer.max_epochs 10 --model_name "state" --dataset_name "ecg_seq" --use_data_augmentation False --state.init_lr 1e-4 --label_key "stent_or_cabg_010_day" --label_file "/nightingale/share/personal/alex.schubert/ACS_benchmark/all_ids_labels_tested_with_covars_all_cath.csv" --checkpoint_path "/nightingale/share/personal/alex.schubert/ACS_benchmark/models/Final_models/benchmark_acs_state_v0/dmaxlwcg/checkpoints/epoch=49-step=1100.ckpt" --batch_size 10

python prediction.py --project_name ACS_benchmark --experiment_name resnet_18_tested_final_vf --trainer.max_epochs 10 --model_name "resnet18_1d" --dataset_name "ecg_seq" --use_data_augmentation False --resnet18_1d.init_lr 1e-4 --label_key "stent_or_cabg_010_day" --label_file "/nightingale/share/personal/alex.schubert/ACS_benchmark/all_ids_labels_tested_with_covars_all_cath.csv" --checkpoint_path "/nightingale/share/personal/alex.schubert/ACS_benchmark/models/Final_models/benchmark_acs_resnet18_1d_final_vf/2vud5fft/checkpoints/epoch=37-step=418.ckpt"

```

Models reported in supplement:

```bash
python prediction.py --project_name ACS_benchmark --experiment_name resnet_6_tested_final_vf --trainer.max_epochs 10 --model_name "resnet6" --dataset_name "ecg_seq" --use_data_augmentation False --ecgomi.init_lr 1e-4 --label_key "stent_or_cabg_010_day" --label_file "/nightingale/share/personal/alex.schubert/ACS_benchmark/all_ids_labels_tested_with_covars_all_cath.csv" --checkpoint_path "/nightingale/share/personal/alex.schubert/ACS_benchmark/models/Final_models/benchmark_acs_ecgomi_final_vf/mqn1y3vw/checkpoints/epoch=40-step=451.ckpt"

python prediction.py --project_name ACS_benchmark --experiment_name resnet_50_tested_final_vf --trainer.max_epochs 10 --model_name "resnet50_1d" --dataset_name "ecg_seq" --use_data_augmentation False --resnet50_1d.init_lr 1e-4 --label_key "stent_or_cabg_010_day" --label_file "/nightingale/share/personal/alex.schubert/ACS_benchmark/all_ids_labels_tested_with_covars_all_cath.csv" --checkpoint_path "/nightingale/share/personal/alex.schubert/ACS_benchmark/models/Final_models/benchmark_acs_resnet50_1d_final_vf/g3fjswg5/checkpoints/epoch=19-step=220.ckpt"
```
