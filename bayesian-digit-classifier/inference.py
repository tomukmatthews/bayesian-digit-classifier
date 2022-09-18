from typing import Optional

import numpy as np
import torch
import training
from model import BayesianConvNet
from PIL import Image, ImageOps
from torchvision import transforms


def load_model(path: str = training.MODEL_PATH):
    model = BayesianConvNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def convert_numpy_array_to_tensor(numpy_image: np.ndarray):
    image = Image.fromarray(np.uint8(numpy_image)).convert("RGB")
    # Invert black and white
    image = ImageOps.invert(image)

    # TODO: Consider cropping whitespace incase user draws small digit
    inference_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            # TODO: Consider normalisation here.
            transforms.ToTensor(),
        ]
    )

    return inference_transforms(image)


def generate_predictions(
    numpy_image: np.ndarray, bayesian_network, n_samples: int = 50
) -> torch.Tensor:
    """Generate predictions of digit class probabilities non-deterministically. Each prediction /
    forward pass of the neural network will generate different predictions, as each forward pass
    involves sampling from the parameter probability distributions.

    Args:
        numpy_image (np.ndarray): The drawn digit.
        n_samples (int, optional): Number of prediction instances (forward passes of the neural
            network) to generate. A single prediction instance returns the probability for each
            class once. Defaults to 50.

    Returns:
        torch.Tensor: 2D tensor, rows are prediction instances,
            columns are the predicted probabilities for each class [0-9].
    """

    tensor = convert_numpy_array_to_tensor(numpy_image)
    # batch dim = 1, colour channels = 1, image size is 28 x 28.
    tensor = tensor.view(1, 1, 28, 28)

    prediction_probas = [torch.exp(bayesian_network(tensor)) for _ in range(n_samples)]
    stacked = torch.stack(prediction_probas)
    return torch.squeeze(stacked)


def calculate_overall_prediction(predictions: torch.Tensor, median_threshold: float = 0.2) -> Optional[int]:
    """Extract a prediction based on the digit with the largest median probability, provided the median
    is larger than a given threshold.

    Args:
        predictions (torch.Tensor): Predictions probabilities for each digit class for n_samples forward 
            passes through the neural network.
        median_threshold (float): Minimum median (over the forward passes) probability for a digit
            classification to be accepted.

    Returns:
        Optional[int]: Digit classification.
    """

    median_digit_probas = torch.median(predictions, 0)
    if torch.max(median_digit_probas.values) >= median_threshold:
        return int(torch.argmax(median_digit_probas.values))
