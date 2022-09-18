import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from blitz.modules.conv_bayesian_layer import BayesianConv2d
from blitz.modules.linear_bayesian_layer import BayesianLinear
from blitz.utils import variational_estimator
from torch import nn
from torchmetrics.classification.accuracy import Accuracy

ACCURACY = Accuracy()


def bayesian_conv2d_unit(
    in_channels: int,
    out_channels: int,
    conv_kernel_size: int,
    max_pool_kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> nn.Sequential:

    bayesian_cnn_unit = nn.Sequential(
        BayesianConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(conv_kernel_size, conv_kernel_size),
            stride=stride,
            padding=padding,
        ),
        nn.MaxPool2d(kernel_size=max_pool_kernel_size),
        nn.LazyBatchNorm2d(),
        nn.ReLU(),
    )
    return bayesian_cnn_unit


@variational_estimator
class BayesianConvNet(pl.LightningModule):
    def __init__(self, colour_channels=1, num_classes=10, learning_rate=0.001):
        super().__init__()

        self.learning_rate = learning_rate

        self.conv_layers = nn.Sequential(
            bayesian_conv2d_unit(
                in_channels=colour_channels,
                out_channels=16,
                conv_kernel_size=4,
                stride=1,
                padding=2,
                max_pool_kernel_size=2,
            ),
            bayesian_conv2d_unit(
                in_channels=16,
                out_channels=32,
                conv_kernel_size=5,
                stride=1,
                padding=2,
                max_pool_kernel_size=2,
            ),
            bayesian_conv2d_unit(
                in_channels=32,
                out_channels=64,
                conv_kernel_size=6,
                stride=2,
                padding=2,
                max_pool_kernel_size=3,
            ),
        )
        self.fc_layers = nn.Sequential(BayesianLinear(64, num_classes), nn.Dropout(0.2), nn.ReLU())

    def forward(self, x):
        x = self.conv_layers(x)
        # Leave batch as first dimension, flatten all other dimensions for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)

    def loss(self, x, y):
        loss = self.sample_elbo(
            inputs=x,
            labels=y,
            criterion=nn.NLLLoss(),
            sample_nbr=3,
            complexity_cost_weight=1 / 50000,
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        train_samples = len(x)
        logits = self.forward(x)
        # loss = F.nll_loss(logits, y)
        loss = self.loss(x, y)
        preds = torch.argmax(logits, dim=1)
        train_acc = ACCURACY(preds, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_acc", train_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log(
            "train_samples",
            float(train_samples),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        val_samples = len(x)
        logits = self.forward(x)
        # loss = F.nll_loss(logits, y)
        loss = self.loss(x, y)
        preds = torch.argmax(logits, dim=1)
        val_acc = ACCURACY(preds, y)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_acc", val_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_samples", float(val_samples), prog_bar=True, on_step=False, on_epoch=True)
