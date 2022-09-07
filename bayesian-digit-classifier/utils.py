import numpy as np
import torch
import train
from model import BayesianConvNet
from PIL import Image, ImageOps


def load_model(path: str = train.PATH):
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
