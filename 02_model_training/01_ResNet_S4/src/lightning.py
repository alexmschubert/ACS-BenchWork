import lightning.pytorch as pl
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import torchvision
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
        
        self.covars = covars
        
        if self.num_classes <= 2:
            self.loss = nn.BCEWithLogitsLoss()  # Binary classification loss
        else:
            self.loss = nn.CrossEntropyLoss()   # Multiclass classification loss
        
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
                x, cov, y = batch["x"], batch["cov"], batch["y"]

            return x, cov.to(torch.long).squeeze(), y.to(torch.long).view(-1)
        
        else:
            if isinstance(batch, list):
                x, y = batch[0], batch[1]
            else:
                assert isinstance(batch, dict)
                x, y = batch["x"], batch["y"]

            return x, y.to(torch.long).view(-1)

    def training_step(self, batch, batch_idx):
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
        
        self.log("train_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.training_outputs = []

    def on_validation_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.validation_outputs])
        y = torch.cat([o["y"] for o in self.validation_outputs])
        
        
        if self.num_classes == 2:
            probs = torch.sigmoid(y_hat)  # For binary, we use sigmoid to get the probabilities
        else:
            probs = F.softmax(y_hat, dim=-1)  # For multiclass, softmax gives probabilities
        
        self.log("val_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.validation_outputs = []

    def on_test_epoch_end(self):
        y_hat = torch.cat([o["y_hat"] for o in self.test_outputs])
        y = torch.cat([o["y"] for o in self.test_outputs])
            
        if self.num_classes == 2:
            probs = torch.sigmoid(y_hat)  # For binary, we use sigmoid to get the probabilities
        else:
            probs = F.softmax(y_hat, dim=-1)  # For multiclass, softmax gives probabilities

        self.log("test_auc", self.auc(probs, y.view(-1)), sync_dist=True, prog_bar=True)
        self.test_outputs = []

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.init_lr, betas = (0.9,0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor':'val_loss'}} 


#######################################
########### ResNet-6 Model ###########
#######################################

class ResNet_6(Classifer):
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
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x, cov=None):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = out + self.shortcut(x)  # Residual connection
        out = self.relu(out)
        return out

#####################################
########### ECG-OMI Model ###########
#####################################

class ECGOMI(Classifer):
    """
    Architecture:
      conv (stride 2) -> 3 residual blocks ->
      conv (stride 25) -> 3 residual blocks ->
      flatten -> 2 FC layers
      
    Feature Extraction ~60k params  
    Classification ~150k params

    Input is assumed to be (batch_size, 3, 5000).
    """
    def __init__(self, num_classes=1, init_lr=1e-4, covars=[], input_length=5000):
        super().__init__(num_classes=num_classes, init_lr=init_lr, covars=covars)
        
        # --- First Conv Layer ---
        # in_channels=3, out_channels=24, kernel_size=7, stride=2 to reduce length
        self.conv1 = nn.Conv1d(
            in_channels=3,
            out_channels=24,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=True
        )
        self.bn1 = nn.BatchNorm1d(24)
        
        # --- 3 Residual Blocks (24 channels) ---
        self.res_blocks_24 = nn.Sequential(*[ResidualBlock_omi(channels=24, kernel_size=3) for _ in range(3)])
        
        # --- Second Conv Layer ---
        # in_channels=24, out_channels=48, kernel_size=7, stride=25 to reduce length from ~2500 to ~100.
        self.conv2 = nn.Conv1d(
            in_channels=24,
            out_channels=48,
            kernel_size=7,
            stride=25,
            padding=3,
            bias=True
        )
        self.bn2 = nn.BatchNorm1d(48)
        
        # --- 3 Residual Blocks (48 channels) ---
        self.res_blocks_48 = nn.Sequential(*[ResidualBlock_omi(channels=48, kernel_size=3) for _ in range(3)])
        
        # After the second conv layer:
        #   Input length = 5000 → after conv1 (stride=2) → ~2500 → after conv2 (stride=25) → ~100.
        # So final feature map shape is (batch, 48, 100) with 4800 features.
        
        # --- Classification ---
        flattened_features = 48 * 100  # 4800 features after flattening.
        hidden_dim = int(150000 // (4800 + 1))  # Approximately 31.
        self.fc1 = nn.Linear(flattened_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Input shape: (batch_size, 3, 5000)
        """
        # First conv layer with stride 2: output shape (batch, 24, ~2500)
        out = F.relu(self.bn1(self.conv1(x)))
        # 3 residual blocks at 24 channels (preserve dimension)
        out = self.res_blocks_24(out)
        # Second conv layer with stride 25: reduces time dimension from ~2500 to ~100; output (batch, 48, 100)
        out = F.relu(self.bn2(self.conv2(out)))
        # 3 residual blocks at 48 channels (preserve dimension)
        out = self.res_blocks_48(out)
        # Flatten: from (batch, 48, 100) to (batch, 4800)
        out = out.flatten(1)
        # Two FC layers
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out

    
class ResidualBlock_omi(nn.Module):
    """
    A standard 1D residual block:
      Conv1d -> BatchNorm -> ReLU -> Conv1d -> BatchNorm -> Skip Connection -> ReLU
    """
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=True)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
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
        dummy_input = torch.zeros(1, input_channels, input_length)  
        x = self.relu(self.bn1(self.initial_conv(dummy_input)))
        x = self.maxpool1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.maxpool2(x)
        flatten_size = x.view(1, -1).size(1)  
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
        dummy_input = torch.zeros(1, input_channels, input_length) 
        x = self.relu(self.bn1(self.initial_conv(dummy_input)))
        x = self.maxpool1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.maxpool2(x)
        flatten_size = x.view(1, -1).size(1) 
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

##############################################
############# State Space Models #############
##############################################

class S4Model(Classifer):
    """
    This implementation follows closely the adaption of the S4 model proposed by Strodthoff, Alcaraz and Haverkamp (2024).

    Strodthoff, N., Lopez Alcaraz, J. M., & Haverkamp, W. (2024). Prospects for artificial intelligence-enhanced
    electrocardiogram as a unified screening tool for cardiac and non-cardiac conditions: an explorative study
    in emergency care. European Heart Journal-Digital Health 

    """

    def __init__(
        self, 
        d_input=1, 
        d_output=1, 
        init_lr=1e-2,
        d_state=64, 
        d_model=512, 
        n_layers=4, 
        dropout=0.2,
        prenorm=False,
        l_max=1024,
        transposed_input=True, 
        bidirectional=True, 
        layer_norm = True, 
        pooling = True, 
    ):
        super().__init__(n_outputs=d_output, init_lr=init_lr)
        self.save_hyperparameters()

        self.prenorm = prenorm

        self.transposed_input = transposed_input
        
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
            self.layer_norm = layer_norm
            if(layer_norm):
                self.norms.append(nn.LayerNorm(d_model))
            else: 
                self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))

        self.pooling = pooling
        # Linear decoder
        if(d_output is None):
            self.decoder = None
        else:
            self.decoder = nn.Linear(d_model, d_output)

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
                z = norm(z.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
            
            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, rate=rate)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)

        x = x.transpose(-1, -2) 

        if(self.pooling):
            # Pooling: average pooling over the sequence length
            x = x.mean(dim=1)

        # Decode the outputs
        if(self.decoder is not None):
            x = self.decoder(x)  
            
        if(not self.pooling and self.transposed_input is True):
            x = x.transpose(-1, -2) 
        return x
    
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