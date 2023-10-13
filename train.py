from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics import YOLO
from tempfile import TemporaryDirectory
import utils.dataloader as dl
from lightning.pytorch import callbacks
import torchmetrics
import lightning.pytorch as pl
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
import yaml
import argparse
from pathlib import Path
import os
import sys
import comet_ml
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

COMETML_KEY = os.getenv('COMETML_KEY')

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
    results = model.train(data=opt.data, epochs=opt.epochs,
                          batch=opt.batch_size)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    if opt.use_comet_ml:
        experiment.end()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
