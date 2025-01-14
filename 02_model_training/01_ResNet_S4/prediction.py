import argparse
import sys
import os
import shutil 
from os.path import dirname, realpath
import torch
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np

sys.path.append(dirname(dirname(realpath(__file__))))
from src.lightning import ResNet_6, ResNet18_1D, ResNet50_1D, S4Model
from src.dataset import ECG_Seq
from lightning.pytorch.cli import LightningArgumentParser
from lightning.pytorch.accelerators import find_usable_cuda_devices
import lightning.pytorch as pl
import torch.nn.functional as F

from sklearn.metrics import roc_curve
from confidenceinterval import roc_auc_score, accuracy_score

# Suppress the specific warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ["WANDB_MODE"] = "offline"

NAME_TO_MODEL_CLASS = {
    "resnet6": ResNet_6,
    "resnet18_1d": ResNet18_1D,
    "resnet50_1d": ResNet50_1D,
    'state': S4Model
}

NAME_TO_DATASET_CLASS = {
    'ecg_seq': ECG_Seq
}


def add_main_args(parser: LightningArgumentParser) -> LightningArgumentParser:

    parser.add_argument(
        "--model_name",
        default="resnet",
        help="Name of model to use. Options include: mlp, cnn, resnet",
    )

    parser.add_argument(
        "--dataset_name",
        default="ecg_img",
        help="Name of dataset to use. Options: pathmnist, nlst"
    )

    parser.add_argument(
        "--use_data_augmentation",
        default=True,
        help="Whether to apply data augmentation to the dataset"
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        help="Which batch size to use during training"
    )

    parser.add_argument(
        "--project_name",
        default="Benchmark_paper",
        help="Name of project for wandb"
    )

    parser.add_argument(
        "--experiment_name",
        default=None,
        help="Name of experiment for wandb"
    )

    parser.add_argument(
        "--monitor_key",
        default="val_auc",
        help="Name of metric to use for checkpointing. (e.g. val_loss, val_acc)"
    )
    
    parser.add_argument(
        "--label_key",
        default="macetrop_pos_or_death_030",
        help="Name of label to predict"
    )
    
    parser.add_argument(
        "--label_file",
        default="/home/ngsci/project/NEJM_benchmark/all_ids_labels_untested.csv",
        help="Path to label dataset"
    )

    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Path to checkpoint to load from. If None, init from scratch."
    )

    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="Whether to train the model."
    )

    return parser

def parse_args() -> argparse.Namespace:
    parser = LightningArgumentParser()
    parser.add_lightning_class_args(pl.Trainer, nested_key="trainer")
    for model_name, model_class in NAME_TO_MODEL_CLASS.items():
        parser.add_lightning_class_args(model_class, nested_key=model_name)
    for dataset_name, data_class in NAME_TO_DATASET_CLASS.items():
        parser.add_lightning_class_args(data_class, nested_key=dataset_name)
    parser = add_main_args(parser)
    args = parser.parse_args()
    return args

def load_from_checkpoint(pl_model, checkpoint_path):
    """ load from checkpoint function that is compatible with S4
    """
    lightning_state_dict = torch.load(checkpoint_path)
    state_dict = lightning_state_dict["state_dict"]
    
    for name, param in pl_model.named_parameters():
        param.data = state_dict[name].data
    for name, param in pl_model.named_buffers():
        param.data = state_dict[name].data

def main(args: argparse.Namespace):
    print(args)
    print("Loading data ..")

    print("Preparing lightning data module (encapsulates dataset init and data loaders)")
    """
        Most the data loading logic is pre-implemented in the LightningDataModule class for you.
        However, you may want to alter this code for special localization logic or to suit your risk
        model implementations
    """

    dataset_args = vars(args[args.dataset_name])  
    dataset_args['use_data_augmentation'] = bool(args.use_data_augmentation)
    dataset_args['use_data_augmentation'] = bool(args.use_data_augmentation)
    dataset_args['batch_size'] = int(args.batch_size)
    dataset_args['label_key'] = str(args.label_key)
    dataset_args['label_file'] = str(args.label_file)

    datamodule = NAME_TO_DATASET_CLASS[args.dataset_name](**dataset_args)

    print("Initializing model")
    if args.checkpoint_path is None:
        model = NAME_TO_MODEL_CLASS[args.model_name](**vars(args[args.model_name]))
    else:
        if args.model_name == 'state':
            model = NAME_TO_MODEL_CLASS[args.model_name](**vars(args[args.model_name]))
            load_from_checkpoint(model, args.checkpoint_path)
        else:
            print(args.model_name)
            model = NAME_TO_MODEL_CLASS[args.model_name].load_from_checkpoint(args.checkpoint_path)

    ##########################################
    ### Calculate outputs
    ##########################################
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    datamodule.prepare_data()
    datamodule.setup()
    #testloader = datamodule.test_dataloader() #In case of validation in heldout sample
    testloader = datamodule.val_dataloader()

    preds, probas = [], []
    curve_ids = []
    label_list = []
    model.eval()
    for batch in tqdm(testloader):
        ecgs, lables, ids = batch['x'], batch['y'], batch['id'] #cov, batch['cov'],
        
        ecgs = ecgs.to(device)
        ecgs = ecgs.float()  # or ecgs.to(dtype=torch.float32)
        pred = model(ecgs)
        proba = F.sigmoid(pred)[:,-1] 
        pred = pred[:,-1]
        
        preds.append(pred.detach().cpu().numpy())
        probas.append(proba.detach().cpu().numpy())
        label_list.append(lables.detach().cpu().numpy())
        curve_ids.append(ids)
    
    curve_ids = [item for sublist in curve_ids for item in sublist]
    curve_ids = np.array(curve_ids)
    
    test_pred_df = pd.DataFrame({'curve_idx': curve_ids, 
                             'preds': np.concatenate(preds),
                                'probas': np.concatenate(probas),
                                 'label':  np.concatenate(label_list)
                                })

    test_pred_df['split'] = 'test'
    
    auc, ci = roc_auc_score(test_pred_df['label'], test_pred_df['preds'],
                        confidence_level=0.95)
    
    print(f'Test AUC Score: {auc} ({ci[0]}, {ci[1]})')
    
    # Step 2: Calculate the optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(test_pred_df['label'], test_pred_df['preds'])
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)
    optimal_threshold = thresholds[optimal_idx]
    

    # Step 3: Binarize the predictions based on the optimal threshold
    test_pred_df['binary_preds'] = (test_pred_df['preds'] >= optimal_threshold).astype(int)
    
    acc, ci_acc = accuracy_score(test_pred_df['label'], test_pred_df['binary_preds'],
                        confidence_level=0.95)
    
    print(f'Test Accuracy Score: {acc} ({ci_acc[0]}, {ci_acc[1]})')
    
    directory_path = os.path.dirname(args.checkpoint_path)

    test_pred_df.to_csv(f'{directory_path}/{args.model_name}_test_val_train_predictions_tested_v1.csv', index = False)

    print("Done")


if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
