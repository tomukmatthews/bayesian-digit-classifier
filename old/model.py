from PIL import Image
import glob

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader, ConcatDataset, Dataset
from torch import nn
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from torchmetrics import Accuracy
from torchvision.datasets import MNIST

import numpy as np

accuracy = Accuracy()


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir="processed_data/", batch_size=64, img_size=512):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = 512
        self.num_classes = 2
        self.basic_transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        self.train_augmenter_1 = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                # transforms.CenterCrop(500),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(10),
                # transforms.RandomGrayscale(),
                transforms.RandomAffine(translate=(0.05, 0.05), degrees=0),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ]
        )

        self.dims = (img_size, img_size, 1)

    def setup(self, stage=None):
        train_data = datasets.ImageFolder(self.data_dir + "/train", transform=self.basic_transform)
        train_data_1 = datasets.ImageFolder(
            self.data_dir + "/train", transform=self.train_augmenter_1
        )
        self.brain_train = ConcatDataset([train_data, train_data_1])
        self.brain_val = datasets.ImageFolder(
            self.data_dir + "/val", transform=self.basic_transform
        )

        self.brain_train, _ = random_split(self.brain_train, [len(self.brain_train), 0])
        self.brain_val, _ = random_split(self.brain_val, [len(self.brain_val), 0])

        # self.brain_val = datasets.ImageFolder(self.data_dir + "/val", transform=self.basic_transform)

        # all_data = datasets.ImageFolder("data/", transform=self.basic_transform)
        # obsv = len(all_data)
        # train_samples = int(0.8 * obsv)
        # val_samples = obsv - train_samples
        # self.brain_train, self.brain_val = random_split(all_data, [train_samples, val_samples])

    def train_dataloader(self):
        return DataLoader(self.brain_train, batch_size=self.batch_size, num_workers=0, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.brain_val, batch_size=self.batch_size, num_workers=0, shuffle=True)


class BrainTumor(pl.LightningModule):
    def __init__(
        self,
        height,
        width,
        channels,
        num_classes,
        batch_size,
        hidden_size=10,
        learning_rate=6e-3,
    ):

        super().__init__()

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 5, kernel_size=6, stride=3, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 10, kernel_size=5, stride=2),
            nn.MaxPool2d(kernel_size=8, stride=4),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        # self.fc_layers = nn.Sequential(nn.Linear(810, 32), nn.Dropout(p=0.5), nn.ReLU(), nn.Linear(32, num_classes),)
        self.fc_layers = nn.Sequential(nn.Linear(810, num_classes), nn.Dropout(p=0.2), nn.ReLU())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        # x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = self.fc_layers(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        train_samples = len(x)
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        train_acc = accuracy(preds, y)
        self.log("train_acc", train_acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_samples", train_samples, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        val_samples = len(x)
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log("val_acc", acc, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_samples", val_samples, prog_bar=True, on_step=False, on_epoch=True)
        return loss


dm = DataModule(batch_size=128)
model = BrainTumor(*dm.size(), num_classes=dm.num_classes, batch_size=dm.batch_size)
trainer = pl.Trainer(max_epochs=12, progress_bar_refresh_rate=20)

# # Specify a path
PATH = "brain_model.pt"

# Save
save = True
if save:
    trainer.fit(model=model, datamodule=dm)
    torch.save(model.state_dict(), PATH)


# Load
model = BrainTumor(height=512, width=512, channels=1, num_classes=2, batch_size=dm.batch_size)
model.load_state_dict(torch.load(PATH))
model.eval()

inference_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ]
)


def predict(IMG_PATH="data/positive/Y3.jpg", MODEL_PATH=PATH, IMG_DIMS=512):
    model = BrainTumor(
        height=IMG_DIMS,
        width=IMG_DIMS,
        channels=1,
        num_classes=2,
        batch_size=dm.batch_size,
    )
    model.load_state_dict(torch.load(PATH))
    model.eval()

    trans = inference_transforms
    image = Image.open(IMG_PATH)
    input = trans(image)
    input = input.view(1, 1, IMG_DIMS, IMG_DIMS)
    output = model(input)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    print(prediction)


def train_set_accuracy(MODEL_PATH=PATH, IMG_DIMS=512):
    files = []
    for file in glob.glob("processed_data/train/positive/*.jpg"):
        files.append((file, 1))
    for file in glob.glob("processed_data/train/positive/*.tif"):
        files.append((file, 1))
    for file in glob.glob("processed_data/train/negative/*.jpg"):
        files.append((file, 0))

    model = BrainTumor(
        height=IMG_DIMS,
        width=IMG_DIMS,
        channels=1,
        num_classes=2,
        batch_size=dm.batch_size,
    )
    model.load_state_dict(torch.load(PATH))
    model.eval()
    trans = inference_transforms
    corr = 0
    for file in files:
        (image, img_class) = file
        image = Image.open(image)
        input = trans(image)
        input = input.view(1, 1, IMG_DIMS, IMG_DIMS)
        output = model(input)
        prediction = int(torch.max(output.data, 1)[1].numpy())
        if prediction == img_class:
            corr += 1
    print("Accuracy: ", corr / len(files))


train_set_accuracy()
predict(IMG_PATH="processed_data/train/positive/Y52.jpg")

# import gradio as gr


# def predict(inp):
#     inp = Image.open(inp)
#     # inp = Image.fromarray(inp.astype("uint8"))
#     inp = inference_transforms(inp)
#     # inp = transforms.ToTensor()(inp).unsqueeze(0)
#     output = model(input)
#     prediction = int(torch.max(output.data, 1)[1].numpy())
#     labels = ["Negative", "Positive"]
#     return {labels[i]: float(prediction[i]) for i in range(1000)}


# inputs = gr.inputs.Image()
# outputs = gr.outputs.Label(num_top_classes=2)
# gr.Interface(fn=predict, inputs=inputs, outputs=outputs).launch(share=True)
