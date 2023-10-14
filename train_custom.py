from pathlib import Path
import os
import sys
import yaml
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
import torchmetrics
from lightning.pytorch.callbacks import ModelCheckpoint
import utils.dataloader as dl
from tempfile import TemporaryDirectory
from utils.dataloader import load_dataset, get_data_loaders
from utils.network import Network
from utils.general import *

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

COMETML_KEY = os.getenv('COMETML_KEY')

class LitModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.network = Network(num_classes)
        
        # torch.nn.Sequential(
        #     torchvision.models.resnet50(),
        #     nn.Linear(in_features=1000, out_features=512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(in_features=512, out_features=num_classes)
        # )
        
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes)

    def validation_step(self, batch, batch_idx):
        inputs, labels, box = batch
        [outputs, box_pred] = self(inputs)
        cls_loss = self.loss_function(outputs, labels)
        box_loss = F.mse_loss(box_pred, box)
        loss = cls_loss + box_loss
        accuracy = self.accuracy(outputs, labels)
        self.log("val_loss", loss)
        self.log('val_acc', accuracy, prog_bar=True)

    def forward(self, x):
        return self.network(x)

    def training_step(self, batch, batch_idx):
        inputs, labels, box = batch
        [outputs, box_pred] = self(inputs)
        cls_loss = self.loss_function(outputs, labels)
        box_loss = F.mse_loss(box_pred, box)
        loss = cls_loss + box_loss
        self.accuracy(outputs, labels)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, target = batch
        outputs = self(inputs)
        loss = self.loss_function(outputs, target)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


def main(opt):
    data_yaml_path = opt.data
    data_ann = read_yaml(data_yaml_path)        
     
    datasets = load_dataset(data_ann, ROOT)
    data_loaders = get_data_loaders(datasets=datasets, datasets_names=['train'], batch_size=opt.batch_size)
    print(data_loaders['train'])
    
    model = LitModel(data_ann['nc'])
    
    checkpoint_callback = ModelCheckpoint(monitor='val_acc', mode='max', verbose=True)
    trainer = pl.Trainer(accelerator='gpu', max_epochs=opt.epochs, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=data_loaders['train'])

if __name__ == '__main__':
    opt = parse_opt(ROOT)
    main(opt)
