import os
from typing import Union

import cv2
import numpy as np
import torch

from models import LeNet5


@torch.no_grad()
def infer_img(
    img_p: Union[str, os.PathLike],
    model: LeNet5,
    device: str
) -> int:
    model.to(device)
    model.eval()

    img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))[np.newaxis, np.newaxis, ...].astype(np.float32)
    img = torch.as_tensor(img / 255.0).to(device)
    
    outputs = model(img)
    outputs = torch.squeeze(outputs)
    _, pred_cat_id = torch.min(outputs, dim = 0)
    pred_cat_id = pred_cat_id.item()

    return pred_cat_id
