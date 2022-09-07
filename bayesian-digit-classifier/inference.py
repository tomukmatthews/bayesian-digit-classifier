import numpy as np
import torch
import utils
from torchvision import transforms


def predict(numpy_image: np.ndarray, process=False, raw=False):

    if process:
        image = utils.process_image(numpy_image)
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

    model = utils.load_model()

    input = inference_transforms(image)
    # batch dim = 1, colour channels = 1, image size is 28 x 28.
    input = input.view(1, 1, 28, 28)
    prediction_log_probas = model(input)
    prediction_probas = torch.exp(prediction_log_probas)
    if raw:
        return prediction_probas

    prediction = int(torch.max(prediction_probas.data, 1)[1].numpy())
    return prediction_probas, prediction


def bayesian_predict(numpy_image: np.ndarray, process=False, n_samples: int = 10):

    if process:
        image = utils.process_image(numpy_image)
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

    model = utils.load_model()

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
