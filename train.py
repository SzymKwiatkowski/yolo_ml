import argparse
from pathlib import Path
import os
import sys
import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

COMETML_KEY = os.getenv('COMETML_KEY')

import yaml
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import lightning.pytorch as pl
import torchmetrics
from lightning.pytorch import callbacks
import utils.dataloader as dl
from tempfile import TemporaryDirectory
# from sklearn.model_selection import train_test_split

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # Root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=ROOT /
                        'yolov8n.pt', help='initial weights path')
    parser.add_argument('--data', type=str, default=ROOT /
                        'data/data.yaml', help='dataset.yaml path relative to datasets folder')
    parser.add_argument('--epochs', type=int, default=100,
                        help='total training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='total training epochs')
    
    parser.add_argument('--use_comet_ml', type=bool, default=False,
                        help='total training epochs')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    print('Config:\n', opt)
    if opt.use_comet_ml:
        experiment = Experiment(
            api_key=COMETML_KEY,
            project_name="PSD_ML",
            workspace="szymkwiatkowski"
        )
    
    model = YOLO(opt.model)
    results = model.train(data=opt.data, epochs=opt.epochs, batch=opt.batch_size)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    
    if opt.use_comet_ml:
        experiment.end()
    
    
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
