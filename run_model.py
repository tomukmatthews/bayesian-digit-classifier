from torchvision import transforms
from pytorch_lightning import Trainer, callbacks
from data_module import MNISTDataModule
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from PIL import Image, ImageOps
import numpy as np

from model import BayesianConvNet


EARLY_STOP_CALLBACK = EarlyStopping(
    monitor="val_acc",
    stopping_threshold=0.98,
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode="max",
)

PATH = "bayesian_model.pt"


def train_model(save: bool = True, path: str = PATH):
    model = BayesianConvNet()
    mnist_dm = MNISTDataModule()
    trainer = Trainer(max_epochs=12, callbacks=[EARLY_STOP_CALLBACK])
    trainer.fit(model, mnist_dm)
    if save:
        torch.save(model.state_dict(), path)


def load_model(path: str = PATH):
    model = BayesianConvNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# TODO: Choose heuristic for assessing whether an output is a valid prediction - if it's a number.
# TODO: Plot box plot with predictions and error bars etc.


def process_image(numpy_image: np.ndarray):

    image = Image.fromarray(np.uint8(numpy_image)).convert("RGB")
    # Invert black and white
    image = ImageOps.invert(image)
    return image


def inference(numpy_image: np.ndarray, process=False, raw=False):

    if process:
        image = process_image(numpy_image)
        inference_transforms = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                # TODO: Consider normalisation here.
                transforms.ToTensor(),
            ]
        )

    else:
        image = numpy_image
        inference_transforms = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                # TODO: Consider normalisation here.
                # transforms.ToTensor(),
            ]
        )
    
    
    model = load_model()

    input = inference_transforms(image)
    # batch dim = 1, colour channels = 1, image size is 28 x 28.
    input = input.view(1, 1, 28, 28)
    prediction_log_probas = model(input)
    prediction_probas = torch.exp(prediction_log_probas)
    if raw:
        return prediction_probas

    prediction = int(torch.max(prediction_probas.data, 1)[1].numpy())
    return prediction_probas, prediction


def bayesian_inference(numpy_image: np.ndarray, process=False, n_samples: int = 10):

    if process:
        image = process_image(numpy_image)
        inference_transforms = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((28, 28)),
                # TODO: Consider normalisation here.
                transforms.ToTensor(),
            ]
        )

    else:
        image = numpy_image
        inference_transforms = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                # TODO: Consider normalisation here.
                # transforms.ToTensor(),
            ]
        )

    model = load_model()

    input = inference_transforms(image)
    # batch dim = 1, colour channels = 1, image size is 28 x 28.
    input = input.view(1, 1, 28, 28)

    prediction_probas = []
    for _ in range(n_samples):
        prediction_log_probas = model(input)
        prediction_probas.append(torch.exp(prediction_log_probas))
    stacked = torch.stack(prediction_probas)
    out = torch.squeeze(stacked)
    # mean = torch.mean(stacked, axis=0)
    # std = torch.std(stacked, axis=0)
    # print('SDS: ', mean)
    # print('SS: ', std)
    return out
    # return torch.stack(prediction_probas)

    # prediction = int(torch.max(prediction_probas.data, 1)[1].numpy())
    # return prediction_probas, prediction



if __name__ == "__main__":
    train_model(save=False)
