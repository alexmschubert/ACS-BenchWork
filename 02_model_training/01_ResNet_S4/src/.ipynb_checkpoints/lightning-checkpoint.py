import lightning.pytorch as pl
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
#from src.cindex import concordance_index
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve
import math
from src.s42 import S4 as S42

class Classifer(pl.LightningModule):
    def __init__(self, num_classes=2, init_lr=1e-4, covars=[], **kwargs):
        super().__init__()
        self.init_lr = init_lr
        self.num_classes = num_classes
        
        # if len(covars) > 0:
        #     covars_list = json.loads(covars)
        #     self.covars = covars_list 
        # else:
        self.covars = covars

        # Define loss fn for classifier
        # self.loss = nn.CrossEntropyLoss()
        
        if self.num_classes <= 2:
            self.loss = nn.BCEWithLogitsLoss()  # Binary classification loss
        else:
            self.loss = nn.CrossEntropyLoss()   # Multiclass classification loss

        # self.accuracy = torchmetrics.Accuracy(task="binary" if self.num_classes <= 2 else task="multiclass", num_classes=self.num_classes)
        # self.auc = torchmetrics.AUROC(task="binary" if self.num_classes <= 2 else task="multiclass", num_classes=self.num_classes)
        
        self.accuracy = torchmetrics.Accuracy(
            task="binary" if self.num_classes <= 2 else "multiclass", 
            num_classes=self.num_classes if self.num_classes > 2 else None
        )

        self.auc = torchmetrics.AUROC(
            task="binary" if self.num_classes <= 2 else "multiclass", 
            num_classes=self.num_classes if self.num_classes > 2 else None
        )

        self.training_outputs = []
        self.validation_outputs = []
        self.test_outputs = []

    def get_xy(self, batch):
        if len(self.covars) > 0:
            if isinstance(batch, list):
                x, y = batch[0], batch[1]
            else:
                assert isinstance(batch, dict)
                #x, y = batch["x"], batch["y_seq"][:,0]
                x, cov, y = batch["x"], batch["cov"], batch["y"]

            return x, cov.to(torch.long).squeeze(), y.to(torch.long).view(-1)
        
        else:
            if isinstance(batch, list):
                x, y = batch[0], batch[1]
            else:
                assert isinstance(batch, dict)
                #x, y = batch["x"], batch["y_seq"][:,0]
                x, y = batch["x"], batch["y"]

            return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx):
        if len(self.covars) > 0:
            x, cov, y = self.get_xy(batch)

            ## TODO: get predictions from your model and store them as y_hat
            y_hat = self.forward(x, cov)
        else:
            x, y = self.get_xy(batch)

            ## TODO: get predictions from your model and store them as y_hat
            y_hat = self.forward(x)
        
        if y_hat.dim() > 1:
            y_hat = y_hat.squeeze(dim=1)
            
        y = y.float()

        loss = self.loss(y_hat,y)

        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_loss', loss, prog_bar=True)

        ## Store the predictions and labels for use at the end of the epoch
        self.training_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def validation_step(self, batch, batch_idx):
        
        if len(self.covars) > 0:
            x, cov, y = self.get_xy(batch)

            y_hat = self.forward(x, cov)
        else:
            x, y = self.get_xy(batch)

            y_hat = self.forward(x)
        
        if y_hat.dim() > 1:
            y_hat = y_hat.squeeze(dim=1)
            
        y = y.float()
        
        # print(f'y shape: {y.shape}')
        # print(f'y_hat shape: {y_hat.shape}')
        
        loss = self.loss(y_hat,y)

        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        self.log("val_acc", self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.validation_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss

    def test_step(self, batch, batch_idx):
        
        if len(self.covars) > 0:
            x, cov, y = self.get_xy(batch)
            y_hat = self.forward(x, cov)
            
        else:
            x, y = self.get_xy(batch)
            y_hat = self.forward(x)
        
        if y_hat.dim() > 1:
            y_hat = y_hat.squeeze(dim=1)
            
        y = y.float()

        loss = self.loss(y_hat,y)

        self.log('test_loss', loss, sync_dist=True, prog_bar=True)
        self.log('test_acc', self.accuracy(y_hat, y), sync_dist=True, prog_bar=True)

        self.test_outputs.append({
            "y_hat": y_hat,
            "y": y
        })
        return loss
    
    def on_train_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.training_outputs])
        y = torch.cat([o["y"] for o in self.training_outputs])
        
        if self.num_classes == 2:
            probs = torch.sigmoid(y_hat)  # For binary, we use sigmoid to get the probabilities
        else:
            probs = F.softmax(y_hat, dim=-1)  # For multiclass, softmax gives probabilities
        
        # if self.num_classes == 2:
        #     probs = F.softmax(y_hat, dim=-1)[:,-1]
        # else:
        #     probs = F.softmax(y_hat, dim=-1)
        
        self.log("train_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        
        # if self.num_classes == 2:
        #     probs = F.softmax(y_hat, dim=-1)[:,-1]
        # else:
        #     probs = F.softmax(y_hat, dim=-1)
        
        if self.num_classes == 2:
            probs = torch.sigmoid(y_hat)  # For binary, we use sigmoid to get the probabilities
        else:
            probs = F.softmax(y_hat, dim=-1)  # For multiclass, softmax gives probabilities
        
        self.log("val_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])

        # if self.num_classes == 2:
        #     probs = F.softmax(y_hat, dim=-1)[:,-1]
        # else:
        #     probs = F.softmax(y_hat, dim=-1)
            
        if self.num_classes == 2:
            probs = torch.sigmoid(y_hat)  # For binary, we use sigmoid to get the probabilities
        else:
            probs = F.softmax(y_hat, dim=-1)  # For multiclass, softmax gives probabilities

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):
        ## TODO: Define your optimizer and learning rate scheduler here (hint: Adam is a good default)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr, betas = (0.9,0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor':'val_loss'}} #'val_loss'


#####################################
########### ECG-OMI Model ###########
#####################################

class EKGOMI(Classifer):
    def __init__(self, input_channels=3, input_length=5000, channels=32, num_classes=1, init_lr = 1e-4, hidden_dim = 256, block_kernel=7, covars=[]): #channels=16,channels=64 works well
        super().__init__(num_classes=num_classes, init_lr=init_lr, covars=covars)
        
        # First Convolution Layer (Part 1: ~60,000 parameters)
        self.initial_conv = nn.Conv1d(input_channels, channels, kernel_size=50, stride=5, padding=3)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.covars = covars
        
        # First set of residual blocks
        self.res_block1 = self._make_res_block(channels, channels*2, num_blocks=3, stride=3, kernel_size=block_kernel)
        self.conv2 = nn.Conv1d(channels*2, channels*2, kernel_size=25, stride=10, padding=1)
        self.res_block2 = self._make_res_block(channels*2, channels*2, num_blocks=3, stride=3, kernel_size=block_kernel)
        
        # Dynamically calculate the size of the flattened layer
        flatten_input_size = self._get_flatten_size(input_channels, input_length)
        
        # Fully connected layers (Part 2: ~150,000 parameters) as a Sequential object
        self.fc = nn.Sequential(
            nn.Linear(flatten_input_size, hidden_dim),  # Assuming the final output is downsampled to 625 features
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
        if len(self.covars) > 0:
            dim_covar = len(self.covars) + num_classes
            self.covars_cond = nn.Sequential(
            nn.Linear(dim_covar, dim_covar),  # Assuming the final output is downsampled to 625 features
            nn.BatchNorm1d(dim_covar),
            nn.ReLU(inplace=True),
            nn.Linear(dim_covar, num_classes)
        )
    
    def _make_res_block(self, in_channels, out_channels, num_blocks, stride, kernel_size=3):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, kernel_size=kernel_size))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, kernel_size=kernel_size))
        return nn.Sequential(*layers)
    
    def _get_flatten_size(self, input_channels, input_length):
        """ 
        This function performs a forward pass with a dummy input to dynamically calculate 
        the size of the feature vector after the convolutional layers and residual blocks.
        """
        dummy_input = torch.zeros(1, input_channels, input_length)  # Batch size of 1, with input_channels and input_length
        x = self.relu(self.bn1(self.initial_conv(dummy_input)))
        x = self.res_block1(x)
        x = self.conv2(x)
        x = self.res_block2(x)
        flatten_size = x.view(1, -1).size(1)  # Flatten and get the size
        return flatten_size
    
    def forward(self, x, cov=None):
        # Initial convolution
        x = self.relu(self.bn1(self.initial_conv(x)))
        
        # First residual blocks
        x = self.res_block1(x)
        x = self.conv2(x)
        
        # Second residual blocks
        x = self.res_block2(x)
         
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        #covariate conditioning if necessary
        if len(self.covars) > 0:
            x = torch.cat((x, cov), dim=1)
            x = self.covars_cond(x)
        
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        # self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        # self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x, cov=None):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        # out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)  # Residual connection
        out = self.relu(out)
        return out

#######################################
########### ResNet-18 Model ###########
#######################################

class ResNet18_1D(Classifer):
    def __init__(self, input_channels=3, input_length=5000, channels=32, num_classes=1, init_lr=1e-4, hidden_dim=256, block_kernel=7, covars=[]):
        super().__init__(num_classes=num_classes, init_lr=init_lr, covars=covars)
        
        # Initial Convolution Layer
        self.initial_conv = nn.Conv1d(input_channels, channels, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # Max pooling after initial convolution
        self.covars = covars
        
        # Residual Blocks
        self.res_block1 = self._make_res_block(channels, channels, num_blocks=4, stride=1, kernel_size=block_kernel)
        self.res_block2 = self._make_res_block(channels, channels * 2, num_blocks=4, stride=2, kernel_size=block_kernel)
        self.res_block3 = self._make_res_block(channels * 2, channels * 4, num_blocks=4, stride=2, kernel_size=block_kernel)
        self.res_block4 = self._make_res_block(channels * 4, channels * 8, num_blocks=4, stride=1, kernel_size=block_kernel)
        
        # Max pooling before the fully connected layer
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        #self.maxpool2 = nn.AdaptiveAvgPool1d(1)
        
        # Dynamically calculate the size of the flattened layer
        flatten_input_size = self._get_flatten_size(input_channels, input_length)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flatten_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
        if len(self.covars) > 0:
            dim_covar = len(self.covars) + num_classes
            self.covars_cond = nn.Sequential(
                nn.Linear(dim_covar, dim_covar),
                nn.BatchNorm1d(dim_covar),
                nn.ReLU(inplace=True),
                nn.Linear(dim_covar, num_classes)
            )
    
    def _make_res_block(self, in_channels, out_channels, num_blocks, stride, kernel_size=3):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, kernel_size=kernel_size))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, kernel_size=kernel_size))
        return nn.Sequential(*layers)
    
    def _get_flatten_size(self, input_channels, input_length):
        """
        This function performs a forward pass with a dummy input to dynamically calculate
        the size of the feature vector after the convolutional layers and residual blocks.
        """
        dummy_input = torch.zeros(1, input_channels, input_length)  # Batch size of 1, with input_channels and input_length
        x = self.relu(self.bn1(self.initial_conv(dummy_input)))
        x = self.maxpool1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.maxpool2(x)
        flatten_size = x.view(1, -1).size(1)  # Flatten and get the size
        return flatten_size
    
    def forward(self, x, cov=None):
        # Initial convolution
        x = self.relu(self.bn1(self.initial_conv(x)))
        x = self.maxpool1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Max pooling before fully connected layers
        x = self.maxpool2(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Covariate conditioning if necessary
        if len(self.covars) > 0:
            x = torch.cat((x, cov), dim=1)
            x = self.covars_cond(x)
        
        return x
    
#######################################
########### ResNet-50 Model ###########
#######################################

class ResNet50_1D(Classifer):
    def __init__(self, input_channels=3, input_length=5000, channels=64, num_classes=1, init_lr=1e-4, hidden_dim=512, block_kernel=7, covars=[]):
        super().__init__(num_classes=num_classes, init_lr=init_lr, covars=covars)
        
        # Initial Convolution Layer
        self.initial_conv = nn.Conv1d(input_channels, channels, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  # Max pooling after initial convolution
        self.covars = covars
        
        # Residual Blocks
        self.res_block1 = self._make_res_block(channels, channels, num_blocks=8, stride=1, kernel_size=block_kernel)
        self.res_block2 = self._make_res_block(channels, channels * 2, num_blocks=11, stride=1, kernel_size=block_kernel)
        self.res_block3 = self._make_res_block(channels * 2, channels * 4, num_blocks=17, stride=1, kernel_size=block_kernel)
        self.res_block4 = self._make_res_block(channels * 4, channels * 8, num_blocks=8, stride=2, kernel_size=block_kernel)
        
        # Max pooling before the fully connected layer
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)
        
        # Dynamically calculate the size of the flattened layer
        flatten_input_size = self._get_flatten_size(input_channels, input_length)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(flatten_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
        if len(self.covars) > 0:
            dim_covar = len(self.covars) + num_classes
            self.covars_cond = nn.Sequential(
                nn.Linear(dim_covar, dim_covar),
                nn.BatchNorm1d(dim_covar),
                nn.ReLU(inplace=True),
                nn.Linear(dim_covar, num_classes)
            )
    
    def _make_res_block(self, in_channels, out_channels, num_blocks, stride, kernel_size=3):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, kernel_size=kernel_size))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, kernel_size=kernel_size))
        return nn.Sequential(*layers)
    
    def _get_flatten_size(self, input_channels, input_length):
        """
        This function performs a forward pass with a dummy input to dynamically calculate
        the size of the feature vector after the convolutional layers and residual blocks.
        """
        dummy_input = torch.zeros(1, input_channels, input_length)  # Batch size of 1, with input_channels and input_length
        x = self.relu(self.bn1(self.initial_conv(dummy_input)))
        x = self.maxpool1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.maxpool2(x)
        flatten_size = x.view(1, -1).size(1)  # Flatten and get the size
        return flatten_size
    
    def forward(self, x, cov=None):
        # Initial convolution
        x = self.relu(self.bn1(self.initial_conv(x)))
        x = self.maxpool1(x)
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        
        # Max pooling before fully connected layers
        x = self.maxpool2(x)
        
        # Flatten and fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        # Covariate conditioning if necessary
        if len(self.covars) > 0:
            x = torch.cat((x, cov), dim=1)
            x = self.covars_cond(x)
        
        return x

##########################################
########### Bidirectional LSTM ###########
##########################################

class LSTM_Conv(Classifer):
    def __init__(self, input_channels=3, input_length=5000, channels=64, num_classes=1, init_lr=1e-4, hidden_dim=512, lstm_hidden_dim=50, covars=[]):
        super().__init__(num_classes=num_classes, init_lr=init_lr, covars=covars)
        
        # First Convolutional Layers
        self.conv_layers1 = nn.Sequential(
            nn.Conv1d(input_channels, channels, kernel_size=3, stride=1, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels * 2, kernel_size=3, stride=1, padding=7),
            nn.BatchNorm1d(channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # First LSTM Layer
        self.lstm1 = nn.LSTM(input_size=channels * 2, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Second Convolutional Layers
        self.conv_layers2 = nn.Sequential(
            nn.Conv1d(lstm_hidden_dim * 2, channels * 4, kernel_size=3, stride=1, padding=7),
            nn.BatchNorm1d(channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels * 4, channels * 8, kernel_size=3, stride=1, padding=7),
            nn.BatchNorm1d(channels * 8),
            nn.ReLU(inplace=True)
        )
        
        # Second LSTM Layer
        self.lstm2 = nn.LSTM(input_size=channels * 8, hidden_size=lstm_hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        
        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.covars = covars
        if len(self.covars) > 0:
            dim_covar = len(self.covars) + num_classes
            self.covars_cond = nn.Sequential(
                nn.Linear(dim_covar, dim_covar),
                nn.BatchNorm1d(dim_covar),
                nn.ReLU(inplace=True),
                nn.Linear(dim_covar, num_classes)
            )
    
    def forward(self, x, cov=None):
        # First Convolutional Layers
        x = self.conv_layers1(x)
        
        # Reshape for First LSTM Layer
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        x, _ = self.lstm1(x)
        
        # Reshape back to (batch_size, channels, seq_len)
        x = x.permute(0, 2, 1)
        
        # Second Convolutional Layers
        x = self.conv_layers2(x)
        
        # Reshape for Second LSTM Layer
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, channels)
        x, _ = self.lstm2(x)
        
        # Flatten and Fully Connected Layer
        x = x[:, -1, :]  # Take the last time step output
        x = self.fc(x)
        
        # Covariate conditioning if necessary
        if len(self.covars) > 0:
            x = torch.cat((x, cov), dim=1)
            x = self.covars_cond(x)
        
        return x

##########################################
################ ViT 1D ##################
##########################################

class ViT_1D(Classifer):
    def __init__(self, input_channels=3, input_length=5000, num_classes=1, init_lr=1e-4, dim=64, depth=5, heads=8, mlp_dim=128, covars=[]):
        super().__init__(num_classes=num_classes, init_lr=init_lr, covars=covars)
        
        self.patch_size = 25  # Define the patch size for splitting the sequence
        self.num_patches = input_length // self.patch_size
        self.dim = dim
        
        # Linear Projection of Patches
        self.patch_to_embedding = nn.Linear(input_channels * self.patch_size, dim)
        
        # Positional Embedding
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        
        # Transformer Encoder Blocks
        self.transformer = nn.Sequential(*[
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
            for _ in range(depth)
        ])
        
        # Fully Connected Layer for Classification
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        self.covars = covars
        if len(self.covars) > 0:
            dim_covar = len(self.covars) + num_classes
            self.covars_cond = nn.Sequential(
                nn.Linear(dim_covar, dim_covar),
                nn.BatchNorm1d(dim_covar),
                nn.ReLU(inplace=True),
                nn.Linear(dim_covar, num_classes)
            )
    
    def forward(self, x, cov=None):
        batch_size = x.size(0)
        
        # Split input into patches and flatten
        x = x.unfold(2, self.patch_size, self.patch_size).permute(0, 2, 1, 3)  # (batch_size, num_patches, input_channels, patch_size)
        x = x.reshape(batch_size, self.num_patches, -1)  # (batch_size, num_patches, input_channels * patch_size)
        
        # Project patches to embeddings
        x = self.patch_to_embedding(x)  # (batch_size, num_patches, dim)
        
        # Add CLS token and positional embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, dim)
        x = x + self.positional_embedding  # (batch_size, num_patches + 1, dim)
        
        # Transformer Encoder
        x = self.transformer(x)  # (batch_size, num_patches + 1, dim)
        
        # Classification token output
        x = x[:, 0]  # (batch_size, dim)
        x = self.fc(x)  # (batch_size, num_classes)
        
        # Covariate conditioning if necessary
        if len(self.covars) > 0:
            x = torch.cat((x, cov), dim=1)
            x = self.covars_cond(x)
        
        return x
    
# class ResidualBlockSimple(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, dropout, stride=2):
#         super().__init__()
#         #padding = kernel_size // 2
#         padding = (kernel_size - 1) // 2  # This calculation handles both even and odd kernel sizes correctly
#         self.block = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
#             nn.BatchNorm1d(out_channels),
#             nn.ReLU()
#         )
#         self.pool = nn.MaxPool1d(kernel_size=stride, stride=stride)

#     def forward(self, x):
#         residual = x
#         out = self.block(x)

#         if residual.size(2) > out.size(2):
#             residual = residual[:, :, :out.size(2)]

#         out = out + residual  # Add the residual to the output of the block
#         out = self.pool(out)
#         return out


# class EKGResNetSimple(Classifer):
#     def __init__(self, n_outputs, init_lr=1e-4, n_channels=1, n_samples=500, num_rep_blocks=8, kernel_size=16, conv_channels=32, dropout=0.5, 
#                  pool_stride=1, hidden_dims=[256, 128], pool='max', output_names = None, covariate_conditioning=None, **kwargs): #[16,16,32,32,64,64,128,128] #[1,2,1,2,1,2,1,1]
        
#         super().__init__(n_outputs=n_outputs, init_lr=init_lr)
#         self.save_hyperparameters()
#         self.covariate_conditioning = covariate_conditioning

#         self.initial_conv = nn.Sequential(
#             nn.Conv1d(n_channels, conv_channels, kernel_size, padding=kernel_size // 2),
#             nn.BatchNorm1d(conv_channels),
#             nn.ReLU(),
#         )

#         blocks = []
#         for i in range(num_rep_blocks):
#             blocks.append(ResidualBlockSimple(conv_channels,conv_channels, kernel_size, dropout, stride=pool_stride))

#             if (i + 1) % 2 == 0:  # Double the kernel size after every two blocks
#                 kernel_size *= 2

#             pool_stride = 2 if pool_stride==1 else 1

#         self.residual_blocks = nn.Sequential(*blocks)

#         # Dummy input to calculate the output size after the last residual block
#         dummy_input = torch.zeros(1, n_channels, n_samples)
#         dummy_output = self.initial_conv(dummy_input)
#         dummy_output = self.residual_blocks(dummy_output)
#         output_size = dummy_output.size(1) * dummy_output.size(2)

#         # Create the fully connected layers compactly
#         self.fc = nn.Sequential(
#             nn.Linear(output_size, hidden_dims[0]),
#             nn.BatchNorm1d(hidden_dims[0]),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dims[0], hidden_dims[1]),
#             nn.BatchNorm1d(hidden_dims[1]),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dims[1], n_outputs),
#             )

#     def forward(self, x, covars=None):
#         x = self.initial_conv(x)
#         x = self.residual_blocks(x)
#         #x = self.adaptive_pool(x)
#         x = x.view(x.size(0), -1)  # Flatten
#         if self.covariate_conditioning is not None:
#             x = torch.cat([x,covars], dim=1)
#         x = self.fc(x)
#         return x

class Conv1DModel(Classifer):
    def __init__(self, input_channels=3, num_classes=2, num_conv_blocks=4, hidden_dim=512, dropout_prob=0.2, init_lr = 1e-4, covars=[], **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, covars=covars)
        
        
        # Convolutional Blocks
        conv_blocks = []
        for i in range(num_conv_blocks):
            conv_blocks.extend([
                nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(p=dropout_prob)
            ])
            input_channels = hidden_dim  # Update input_channels for subsequent blocks

        # Fully Connected Layers
        fc_layers = [
            nn.Flatten(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(256, num_classes)
        ]

        # Create a sequential model
        self.model = nn.Sequential(
            *conv_blocks,
            nn.AdaptiveMaxPool1d(1),
            *fc_layers
        )
        
        print(self.covars)
        print(covars)
        # Hack for model evaluation
        self.covars = ["age_at_admit", "female"]
        
        print(f"Number of covars: {len(self.covars)}")
        
        if len(self.covars) > 0:
            input_dim = len(self.covars) + self.num_classes
            self.cov_ensemble = nn.Sequential(*[nn.Linear(input_dim, 32), nn.BatchNorm1d(32),nn.ReLU(inplace=True),
                                               nn.Linear(32, self.num_classes)])
            #self.cov_ensemble = nn.Linear(input_dim, self.num_classes)

    def forward(self, x, cov=None):
        x = self.model(x)
        if len(self.covars) > 0:
            x = torch.cat((x, cov), dim=1)
            x = self.cov_ensemble(x)
        return x    

#####################################
######## Models 1D Conv #############
#####################################

class Conv1DModel(Classifer):
    def __init__(self, input_channels=3, num_classes=2, num_conv_blocks=4, hidden_dim=128, dropout_prob=0.2, init_lr = 1e-4, covars=[], **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr, covars=covars)
        
        
        # Convolutional Blocks
        conv_blocks = []
        for i in range(num_conv_blocks):
            conv_blocks.extend([
                nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(p=dropout_prob)
            ])
            input_channels = hidden_dim  # Update input_channels for subsequent blocks

        # Fully Connected Layers
        fc_layers = [
            nn.Flatten(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(128, num_classes)
        ]

        # Create a sequential model
        self.model = nn.Sequential(
            *conv_blocks,
            nn.AdaptiveMaxPool1d(1),
            *fc_layers
        )
        
        print(self.covars)
        print(covars)
        # Hack for model evaluation
        self.covars = ["age_at_admit", "female"]
        
        print(f"Number of covars: {len(self.covars)}")
        
        if len(self.covars) > 0:
            input_dim = len(self.covars) + self.num_classes
            self.cov_ensemble = nn.Sequential(*[nn.Linear(input_dim, 32), nn.BatchNorm1d(32),nn.ReLU(inplace=True),
                                               nn.Linear(32, self.num_classes)])
            #self.cov_ensemble = nn.Linear(input_dim, self.num_classes)

    def forward(self, x, cov=None):
        
        x = self.model(x)
        if len(self.covars) > 0:
            x = torch.cat((x, cov), dim=1)
            x = self.cov_ensemble(x)
        return x

##################################################
## Transformer model
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerModel(Classifer):
    def __init__(self, input_channels=3, num_classes=2, d_model=512, nhead=8, num_encoder_layers=3, dim_feedforward=512, dropout=0.2, max_seq_length=1000, init_lr=1e-4, covars=[]):
        super().__init__()
        self.num_classes = num_classes
        self.init_lr = init_lr
        self.covars = covars
        
        print(covars)
        
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.input_proj = nn.Linear(input_channels, d_model)
        self.fc_out = nn.Linear(d_model, num_classes)
        
        if len(covars) > 0:
            input_dim = len(covars) + num_classes
            self.cov_ensemble = nn.Sequential(
                nn.Linear(input_dim, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True),
                nn.Linear(32, num_classes)
            )

    def forward(self, x, cov=None):
        # Adjust x's shape to [batch_size, seq_len, input_channels] if not already
        x = x.permute(0, 2, 1)  # Assuming x is [batch_size, input_channels, seq_len]
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16) as autocast, torch.backends.cuda.sdp_kernel(enable_flash=False) as disable :
            x = self.input_proj(x)  # Now x is [batch_size, seq_len, d_model]
            x = self.pos_encoder(x)
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)  # Aggregate features across the sequence
            x = self.fc_out(x)
        
        if len(self.covars) > 0 and cov is not None:
            x = torch.cat((x, cov), dim=1)
            x = self.cov_ensemble(x)
        return x

##################################################
## LSTM model

class LSTMModel(Classifer):
    def __init__(self, input_channels=3, num_classes=2, hidden_dim=512, num_layers=2, dropout_prob=0, init_lr=1e-4, covars=[], **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.init_lr = init_lr
        self.covars = covars
        
        print(covars)
        
        # LSTM Layers
        self.lstm = nn.LSTM(input_size=input_channels, 
                            hidden_size=hidden_dim, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout_prob if num_layers > 1 else 0, 
                            bidirectional=False)
        
        
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # Optional covariate ensemble
        if len(self.covars) > 0:
            input_dim = len(self.covars) + num_classes
            self.cov_ensemble = nn.Sequential(
                nn.Linear(input_dim, 32), 
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Linear(32, self.num_classes)
            )
        
    def forward(self, x, cov=None):
        x = x.permute(0, 2, 1)
        # LSTM forward pass
        x, (hn, cn) = self.lstm(x)
        
        # Take the last hidden state
        #x = hn[-1]  # If bidirectional, you might want to concatenate or add the last hidden states of both directions
        x = torch.amax(x, dim=1)
        #x = x.max(dim=1)
        
        # Fully connected layer
        x = self.fc(x)
        
        # Covariate ensemble
        if len(self.covars) > 0 and cov is not None:
            x = torch.cat((x, cov), dim=1)
            x = self.cov_ensemble(x)
        return x

##############################################
############# State Space Models #############
##############################################

class S4Model(Classifer):

    def __init__(
        self, 
        d_input=1, # None to disable encoder
        d_output=1, # None to disable decoder 
        init_lr=1e-2,
        d_state=64, #MODIFIED: N
        d_model=512, #MODIFIED: H
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        l_max=1024,
        transposed_input=True, # behaves like 1d CNN if True else like a RNN with batch_first=True
        bidirectional=True, #MODIFIED
        layer_norm = True, # MODIFIED
        pooling = True, # MODIFIED
    ):
        super().__init__(n_outputs=d_output, init_lr=init_lr)
        self.save_hyperparameters()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.transposed_input = transposed_input
        
        # MODIFIED TO ALLOW FOR MODELS WITHOUT ENCODER
        if(d_input is None):
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Conv1d(d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)
        
        print(l_max)
        
        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S42(
                    d_state=d_state,
                    l_max=l_max,
                    d_model=d_model, 
                    bidirectional=bidirectional,
                    postact='glu',
                    dropout=dropout, 
                    transposed=True,
                ))
            #MODIFIED TO ALLOW BATCH NORM MODELS
            self.layer_norm = layer_norm
            if(layer_norm):
                self.norms.append(nn.LayerNorm(d_model))
            else: #MODIFIED
                self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        self.pooling = pooling
        # Linear decoder
        # MODIFIED TO ALLOW FOR MODELS WITHOUT DECODER
        if(d_output is None):
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model, d_output)

    #MODIFIED
    def forward(self, x, rate=1.0):
        """
        Input x is shape (B, d_input, L) if transposed_input else (B, L, d_input)
        """
        x = self.encoder(x)  # (B, d_input, L) -> (B, d_model, L) if transposed_input else (B, L, d_input) -> (B, L, d_model)

        if(self.transposed_input is False):
            x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                # MODIFIED
                z = norm(z.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
            
            # Apply S4 block: we ignore the state input and output
            # MODIFIED
            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                # MODIFIED
                x = norm(x.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)

        x = x.transpose(-1, -2) # (B, d_model, L) -> (B, L, d_model)

        # MODIFIED ALLOW TO DISABLE POOLING
        if(self.pooling):
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

        # Decode the outputs
        if(self.decoder is not None):
            x = self.decoder(x)  # (B, d_model) -> (B, d_output) if pooling else (B, L, d_model) -> (B, L, d_output)
            
        if(not self.pooling and self.transposed_input is True):
            x = x.transpose(-1, -2) # (B, L, d_output) -> (B, d_output, L)
        return x
    
    #NEW, check if this fixes my error
    def load_state_dict(self, state_dict, strict=True):
        # Clone the problematic tensors to avoid memory overlap issues
        if 's4_layers.0.kernel.kernel.B' in state_dict:
            state_dict['s4_layers.0.kernel.kernel.B'] = state_dict['s4_layers.0.kernel.kernel.B'].clone()
        if 's4_layers.0.kernel.kernel.P' in state_dict:
            state_dict['s4_layers.0.kernel.kernel.P'] = state_dict['s4_layers.0.kernel.kernel.P'].clone()
        if 's4_layers.0.kernel.kernel.w' in state_dict:
            state_dict['s4_layers.0.kernel.kernel.w'] = state_dict['s4_layers.0.kernel.kernel.w'].clone()

        # Call the default implementation which now handles the cloned state
        super().load_state_dict(state_dict, strict=strict)


##############################################
######## Models ECG_IMG ######################
##############################################
    
class ResNet(Classifer):
    def __init__(self, input_dim=256*256*3, num_classes=2, init_lr = 1e-4, pretrained=True, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters() 

        self.network = torchvision.models.resnet18(pretrained=pretrained)
        self.network.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.network.fc = nn.Linear(512,num_classes)
    
    def forward(self, x):
        return self.network(x)

class ViT(Classifer):
    def __init__(self, input_dim=256*256*3, num_classes=2, init_lr = 1e-4, pretrained=True, **kwargs):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters() 
        
        if pretrained:
            self.network = torchvision.models.vit_b_16(weights='DEFAULT')
        else:
            self.network = torchvision.models.vit_b_16(weights='DEFAULT')
        self.network.heads.head = nn.Linear(768,num_classes)
        # self.network.avgpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        # self.network.fc = nn.Linear(512,num_classes)
        # print(self.network)
    
    def forward(self, x):
        return self.network(x)

class CNN(Classifer):
    def __init__(self, img_dim=256, in_channels = 3, n_filters=128, num_layers=1, num_classes=2, 
                 use_bn=True, init_lr = 1e-3, 
                 pool_every=1, conv_kernel_size =5,
                 pool_kernel_size =2, fc_hidden_size=128, **kwargs):
        
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters() 

        self.n_filters = n_filters
        self.use_bn = use_bn

        layers = []
        in_channels = 3  
        for i in range(num_layers):
            pad = (conv_kernel_size - 1) // 2 # pad such that we maintain the image dimension
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, self.n_filters, kernel_size=conv_kernel_size, stride=1, padding=pad),
                nn.ReLU(),
                )
            if self.use_bn:
                conv_layer.add_module('batch_norm', nn.BatchNorm2d(n_filters))
            layers.append(conv_layer)
            if i % pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
            
            in_channels = n_filters  # Update the input channels for the next layer
        
        layers.append(nn.AdaptiveMaxPool2d(output_size=(1, 1)))
        
        self.layers = nn.Sequential(*layers)
#         final_size = img_dim // (pool_kernel_size ** (num_layers//pool_every))

#         if final_size == 0:
#             raise ValueError("The final size of the output feature map is 0, which is invalid. Adjust the number of layers or the size of the input image.")

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_filters, fc_hidden_size),  
            nn.ReLU(),
            nn.Linear(fc_hidden_size, num_classes),  
        )

    def forward(self, x):
        x = self.layers(x)
        return self.output(x)


class CustomCNN(Classifer):
    def __init__(self, num_layers=4, num_classes=2, init_lr = 1e-4):
        super().__init__(num_classes=num_classes, init_lr=init_lr)
        self.save_hyperparameters() 

        # Validate the number of layers
        if num_layers < 1:
            raise ValueError("Number of layers must be at least 1.")

        layers = []

        # Initial input channels (for RGB images)
        in_channels = 3

        # Create convolutional layers
        for i in range(num_layers):
            out_channels = 32 * (2 ** i)  # Increasing the number of channels
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ]
            in_channels = out_channels

        # Flatten the output from convolutional layers
        self.conv_layers = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        # Calculate the size of the flattened features
        # Assuming square input images and that each max pool layer halves the dimensions
        linear_input_size = in_channels * (256 // 2**num_layers)**2

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(linear_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
    