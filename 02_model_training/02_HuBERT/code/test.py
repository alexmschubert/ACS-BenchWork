import pandas as pd
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from dataset import ECGDataset
from tqdm import tqdm
from loguru import logger
import argparse
import numpy as np
from torchmetrics.classification import MultilabelF1Score as F1_score
from torchmetrics.classification import MultilabelRecall as Recall
from torchmetrics.classification import MultilabelPrecision as Precision
from torchmetrics.classification import MultilabelSpecificity as Specificity
from torchmetrics.classification import MultilabelAUROC as AUROC
from torchmetrics.classification import MulticlassRecall, MulticlassSpecificity
from torcheval.metrics import MultilabelAUPRC as AUPRC
from torcheval.metrics import MulticlassAUROC, MulticlassAccuracy, MulticlassAUPRC
from typing import Iterable
from hubert_ecg import HuBERTECG as HuBERT
from hubert_ecg_classification import HuBERTForECGClassification as HuBERTClassification
from metrics import CinC2020


def random_crop(ecg, crop_size=500):
    '''
    Performs time-aligned random crop of the input ECG signals in the batch.
    Useful for test-time augmentation.
    
    in : BS, 12 * L
    out : BS, 12 * crop_size
    ---
    Note: crop_size should be 2500 / downsampling_factor
    
    '''
    
    batch_size = ecg.size(0)
    ecg = ecg.view(batch_size, 12, -1)
    new_ecg = torch.zeros(batch_size, 12, crop_size).to(ecg.device)  
    for i in range(batch_size):
        start = np.random.randint(0, ecg.size(-1) - crop_size)
        new_ecg[i] = ecg[i, :, start:start+crop_size]
    return new_ecg.view(batch_size, -1)


def test(args, model : nn.Module, metrics : Iterable[nn.Module]):
    
    #fixing seed
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    testset = ECGDataset(
        path_to_dataset_csv = args.path_to_dataset_csv,
        ecg_dir_path = args.ecg_dir_path,
        pretrain = False,
        encode = False,
        random_crop = False,
        return_full_length = args.tta,
        downsampling_factor=args.downsampling_factor,
        label_start_index=args.label_start_index,
    )
    
    dataloader = DataLoader(
        testset,
        collate_fn=testset.collate,
        num_workers=0,
        batch_size = args.batch_size,
        shuffle=False,
        sampler=None,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info("Test set and dataloader have been created")
    
    model.to(device)
    model.eval()
        
    for name, metric in metrics.items():
        metric.to(device)
        
    if args.challenge_metric and args.task == 'multi_label':
        cinc2020_metric = CinC2020(testset.diagnoses_cols, verbose=False)
        cinc2020_metric.to(device)
    
    logger.info("Start testing...")
    
    ### TESTING LOOP ###
    filenames, probs_pos, probs_neg, true_labels = [], [], [], []
    for ecg, _, labels, path in tqdm(dataloader, total=len(dataloader)):
        
        ecg = ecg.to(device) # BS x 12 * L
        labels = labels.squeeze().to(device)
        
        ecgs = [random_crop(ecg, 2500//args.downsampling_factor) for _ in range(args.n_augs)] if args.tta else [ecg] # [BS x 12 * crop_size] * N_AUGS
        
        probs = []
        for aug_ecg in ecgs: # iterate over augmented batches
            # forward with a single augmented batch
            with torch.no_grad():
                out = model(aug_ecg, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True)
                if isinstance(model, HuBERT):
                    logits = model.logits(out['last_hidden_state']).transpose(1, 2)
                elif isinstance(model, HuBERTClassification):
                    logits = out[0]
            # end of forward
                    
            probs.append(torch.sigmoid(logits) if args.task == 'multi_label' else torch.softmax(logits, dim=-1)) # [BS x num_labels] * N_AUGS
        
        # average probs over augmented batches for tta
        probs = torch.stack(probs).mean(dim=0) if args.tta_aggregation == 'mean' else torch.stack(probs).max(dim=0).values
        
        if args.challenge_metric:
            preds = (probs > 0.5).float() # binary predictions
            cinc2020_metric.update(preds, labels)
        
        labels = labels.long() # metrics expect long labels, not float
        for name, metric in metrics.items():
            metric.update(probs, labels) # compute metric per batch
        
        # Move data to CPU and convert to NumPy
        probs_np = probs.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Append data to the lists
        filenames.extend(path)  # Add all filenames in the current batch
        probs_pos.extend(probs_np[:, 0])  # Add positive class probabilities
        probs_neg.extend(probs_np[:, 1])  # Add negative class probabilities
        true_labels.extend(labels_np[:, 0])
    
            
    ### END OF TESTING LOOP ###
    
    # Create a DataFrame from collected data
    pred_df = pd.DataFrame({
        'Filename': filenames,
        'Probability_Pos': probs_pos,
        'Probability_Neg': probs_neg,
        'True_Label': true_labels
    })
    
    print(pred_df['True_Label'].sum())
    
    performance = pd.DataFrame(columns=testset.diagnoses_cols, index=list(metrics.keys()))
    
    for name, metric in metrics.items():
        score = metric.compute() # compute metric over all test set
        print(f"{name} = {score}, macro: {score.mean()}")
        performance.loc[name] = score.cpu().numpy() if args.task == 'multi_label' else score.mean().cpu().numpy()
        
    if args.challenge_metric and args.task == 'multi_label':
        score = cinc2020_metric.compute()
        print(f"CinC2020 = {score}")
        with open("cinc2020.txt", 'a') as f:
            f.write(f"{args.save_id} : {score}\n")
        
            
        
    if args.save_id is not None:
        if args.tta_aggregation == 'max':
            performance.to_csv(f"./performance/performance_{args.save_id}_max.csv")
            pred_df.to_csv(f"./performance/preds_{args.save_id}_max.csv")
        else:
            performance.to_csv(f"./performance/performance_{args.save_id}.csv")
            pred_df.to_csv(f"./performance/preds_{args.save_id}.csv")
        logger.info(f"Performance metrics saved")
            
    logger.info("End of testing.")
    
if __name__ == "__main__":
    
    ### PARSING ARGUMENTS ###
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("path_to_dataset_csv", type=str, help="Path to dataset csv file")
    
    parser.add_argument("ecg_dir_path", type=str, help="Path to directory with ecg files")
    
    parser.add_argument("batch_size", type=int, help="Batch size")

    parser.add_argument("model_path", type=str, help="Path to a model checkpoint that is to be tested")
    
    parser.add_argument("--downsampling_factor", type=int, help="Downsampling factor")
    
    parser.add_argument("--label_start_index", default=3, type=int, help="Label start index in dataframe. Default 3")    
    
    parser.add_argument("--save_id", type=str, default=None, help="The id to use for saving the csv file of performance metrics")
    
    parser.add_argument("--tta", default=False, action="store_true", help="Whether to use test time augmentation")
    
    parser.add_argument("--tta_aggregation", type=str, default='mean', choices=['mean', 'max'], help="Aggregation method for tta")
    
    parser.add_argument("--n_augs", type=int, default=3, help="Number of augmentations")
    
    parser.add_argument("--challenge_metric", default=False, action="store_true", help="Whether to compute CinC2020 metric")
    
    parser.add_argument("--task", type=str, default='multi_label', choices=['multi_label', 'multi_class', 'regression'], help="Task type")
    
    args = parser.parse_args()
    
    
    # LOADING FINETUNED MODEL FOR INFERENCE
        
    logger.info(f"Loading finetuned model from {args.model_path.split('/')[-1]}")
    
    cpu_device = torch.device('cpu')
    
    checkpoint = torch.load(args.model_path, map_location = cpu_device)
    
    config = checkpoint["model_config"]

    pretrained_hubert = HuBERT(config)

    keys = list(checkpoint['model_state_dict'].keys())    
    num_labels = checkpoint['finetuning_vocab_size']

    if checkpoint['linear']:
        classifier_hidden_size = None
    elif checkpoint['use_label_embedding']:
        classifier_hidden_size = None
    else:
        classifier_hidden_size = checkpoint['model_state_dict'][keys[-2]].size(-1)
        
    hubert = HuBERTClassification(
        pretrained_hubert,
        num_labels=num_labels,
        classifier_hidden_size=classifier_hidden_size,
        use_label_embedding=checkpoint['use_label_embedding'])
    
    hubert.load_state_dict(checkpoint['model_state_dict'], strict=False) # strict false prevents errors when trying to match mask token key
    
    logger.info(f"Loaded finetuned model")
    
    task2metric = {
        'multi_label' : {
            'test_f1_score' : F1_score(num_labels=num_labels, average=None), 
            'test_recall' : Recall(num_labels=num_labels, average=None),
            'test_precision' : Precision(num_labels=num_labels, average=None),
            'test_specificity' : Specificity(num_labels=num_labels, average=None),
            'test_auroc' : AUROC(num_labels=num_labels, average=None), 
            'test_auprc' : AUPRC(num_labels=num_labels, average=None)
                         },
        
        'multi_class' : {
            'test_accuracy' : MulticlassAccuracy(num_classes=num_labels),
            'test_auroc' : MulticlassAUROC(num_classes=num_labels),
            'test_recall' : MulticlassRecall(num_classes=num_labels),
            'test_specificity' : MulticlassSpecificity(num_classes=num_labels),
            'test_auprc' : MulticlassAUPRC(num_classes=num_labels)
                        },
        
        'regression' : {}
                         
    }
    
    metrics = task2metric[args.task]

    ### START TESTING ###
    
    test(args, hubert, metrics) 