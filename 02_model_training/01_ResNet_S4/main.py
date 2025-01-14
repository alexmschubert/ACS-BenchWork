import os
import argparse
import sys
from os.path import dirname, realpath
import warnings
import torch

sys.path.append(dirname(dirname(realpath(__file__))))
from src.lightning import ResNet_6, ResNet18_1D, ResNet50_1D, S4Model
from src.dataset import ECG_Seq
from lightning.pytorch.cli import LightningArgumentParser
import lightning.pytorch as pl

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
        default=64,
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
    
    # Initialize wandb logger
    logger = pl.loggers.WandbLogger(project=args.project_name, 
                                    name=args.experiment_name, 
                                    entity="alexander_schubert")
    
    
    # Log configuration to wandb
    logger.experiment.config.update(vars(args)) 

    print("Preparing lightning data module (encapsulates dataset init and data loaders)")

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
            model = NAME_TO_MODEL_CLASS[args.model_name].load_from_checkpoint(args.checkpoint_path)

    print("Initializing trainer")
    args.trainer.accelerator = 'auto'
    args.trainer.logger = logger
    args.trainer.precision = "bf16-mixed" 
    args.trainer.default_root_dir = "model_checkpoints/{dataset}-{model}".format(dataset=args.dataset_name, model=args.model_name) # for saving model checkpoints

    args.trainer.callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor=args.monitor_key,
            mode='min' if "loss" in args.monitor_key else "max",
            save_last=False,# True
        )]

    trainer = pl.Trainer(**vars(args.trainer))
    print('trainer precision', trainer.precision)
    trainer.log_every_n_steps = 10

    if args.train:
        print("Training model")
        trainer.fit(model, datamodule)

    print("Best model checkpoint path: ", trainer.checkpoint_callback.best_model_path)

    print("Evaluating model on validation set")
    trainer.validate(model, datamodule)

    print("Evaluating model on test set")
    trainer.test(model, datamodule)

    print("Done")


if __name__ == '__main__':
    __spec__ = None
    args = parse_args()
    main(args)
