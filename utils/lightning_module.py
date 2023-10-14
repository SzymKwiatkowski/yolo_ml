import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torchmetrics
from utils.network import Network, ResnetBaseNetwork

class LitModel(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.network = Network(num_classes)
        
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
