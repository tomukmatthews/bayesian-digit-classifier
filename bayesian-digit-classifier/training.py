from typing import Optional

import torch
from data_module import MNISTDataModule
from model import BayesianConvNet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

EARLY_STOP_CALLBACK = EarlyStopping(
    monitor="val_acc",
    stopping_threshold=0.98,
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode="max",
)

PATH = "bayesian-digit-classifier/trained_models/bayesian_model.pt"


def train_model(save_path: Optional[str] = PATH):
    model = BayesianConvNet()
    mnist_dm = MNISTDataModule()
    trainer = Trainer(max_epochs=10, callbacks=[EARLY_STOP_CALLBACK])
    trainer.fit(model, mnist_dm)
    if save_path:
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train_model(save_path=PATH)
