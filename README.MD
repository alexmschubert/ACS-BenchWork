![Banner](assets/ACS_leaderboard_banner.png)

# ACS-BenchWork

This repository contains code and data for the paper:

**Diagnosis of Acute Coronary Syndrome using ECG Waveforms: A Machine Learning Framework and Benchmark Dataset**  
*Alexander Schubert, Ziad Obermeyer*

[Model weights available here](https://huggingface.co/alex-schubert/ACS-benchmark)

## Summary

Electrocardiograms (ECGs) are central to the diagnosis of acute coronary syndromes (ACS). Classical ECG features such as ST-elevation or depression, discovered over 100 years ago, remain widely used to diagnose ACS—despite mounting evidence they are neither sensitive nor specific. Artificial intelligence (AI) could improve ECG diagnosis of ACS. But existing AI-ready datasets contain only cardiologists’ judgments on whether classical features are present in the ECG, not patient health outcomes related to ACS.

In this work, we introduce ACS-BenchWork, a new public dataset in which ECG waveforms from emergency department (ED) visits are labeled with a patient outcome related to ACS: Results of cardiac catheterization within 10 days of the visit, and specifically whether an intervenable lesion was identified. We benchmark both classical ECG features and several machine learning methods on this dataset, showcasing significant potential for AI-based improvements in diagnosing ACS. Our goal is to provide a robust, openly accessible evaluation framework to assess future progress in this task, akin to benchmark datasets in other areas of machine learning.

The dataset is freely available via the [Nightingale Open Science platform](https://docs.ngsci.org/), and we have established a Hugging Face leaderboard that allows researchers to evaluate their models on hidden holdout ECG data. We aim for this to fosters collaborative progress by enabling direct comparison of different approaches for ACS detection.

## Contents

1. **01_data_processing**  
   Contains Python scripts for processing raw ECG data (both 2.5-second and 10-second segments) and generating tabular datasets.

2. **02_model_training**  
   Includes multiple approaches for training machine learning models:
   - `01_ResNet_S4` for training and inference with a ResNet-based model. The S4 model architecture is based on work by [Strodthoff et al.](https://github.com/AI4HealthUOL/ECG-MIMIC)
   - `02_HuBERT` for training and finetuning a HuBERT-based approach. The code is based on the repository from [Coppola et al.](https://github.com/Edoar-do/HuBERT-ECG/tree/master/code) 
   - A `create_logistic_regression_baseline.ipynb` notebook to demonstrate a simpler logistic regression model.

3. **03_model_evaluation**  
   Jupyter notebooks for evaluating model performance, generating descriptive statistics, and visualizing results.

## Getting Started

1. **Clone the Repository**

    ```bash
    git clone https://github.com/YourUsername/ACS-BenchWork.git
    cd ACS-BenchWork
    ```

2. **Install Dependencies**

    We recommend creating a virtual environment.

    ```bash
    conda create --name acs_env python=3.8
    conda activate acs_env
    pip install -r requirements.txt
    ```

3. **Data preparation**

    Run scripts in `01_data_processing/` to process and prepare the raw ECGs ans structured patient data.

4. **Model Training**

    - **S4 and ResNet models**: Navigate to `02_model_training/01_ResNet_S4` and run `main.py` to start training. Follow the instructions in training_and_prediction_commands.md for detailed usage.
    - **HuBERT**: Navigate to `02_model_training/02_HuBERT/code` and use the commands in `finetune.sh` or `test.sh` to finetune/generate predictions for HuBERT-based models.
    - **Logistic Regression Baseline**: Open the `create_logistic_regression_baseline.ipynb` notebook in `02_model_training/` for a simpler baseline approach.

5. **Model Evaluation**
    Use the Jupyter notebooks in `03_model_evaluation/` to:

    - Generate descriptive statistics of your dataset.
    - Analyze the performance of each model.
    - Compare with classical ECG features.


## Data access

Data for this project is hosted on the [Nightingale Open Science](https://docs.ngsci.org/) cloud platform, allowing researchers to train or finetune their models in a standard Amazon Web Services environment. Researchers receive access to the full public dataset (including tested and untested patients), along with holdout ECG waveforms that lack labels or additional metadata. Models can be applied to these holdout waveforms to generate a CSV file of ECG IDs and predictions, which is then uploaded to the Hugging Face competition platform **[LINK TO BE PROVIDED]**. The platform evaluates the predictions against hidden labels, providing performance metrics for both ACS and MACE tasks.

## References

```bibtex
@article{mullainathan2022solving,
  title={Solving medicine’s data bottleneck: Nightingale Open Science},
  author={Mullainathan, Sendhil and Obermeyer, Ziad},
  journal={Nature Medicine},
  volume={28},
  number={5},
  pages={897--899},
  year={2022},
  publisher={Nature Publishing Group US New York}
}

@article{coppola2024hubert,
  title={HuBERT-ECG: a self-supervised foundation model for broad and scalable cardiac applications},
  author={Coppola, Edoardo and Savardi, Mattia and Massussi, Mauro and Adamo, Marianna and Metra, Marco and Signoroni, Alberto},
  journal={medRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press},
  doi={https://doi.org/10.1101/2024.11.14.24317328}
}

@article{strodthoff2024prospects,
  title={Prospects for artificial intelligence-enhanced electrocardiogram as a unified screening tool for cardiac and non-cardiac conditions: an explorative study in emergency care},
  author={Strodthoff, Nils and Lopez Alcaraz, Juan Miguel and Haverkamp, Wilhelm},
  journal={European Heart Journal-Digital Health},
  pages={ztae039},
  year={2024},
  publisher={Oxford University Press UK}
}
```


