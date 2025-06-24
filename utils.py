import os
from typing import Union

import cv2
import numpy as np


def mnist_cvs_to_imgfolder(
    csv_p: Union[str, os.PathLike],
    export_dir: Union[str, os.PathLike],
) -> None:
    imgs = []
    cat_ids = []

    with open(csv_p, "r") as f:
        for l in f:
            l = l.strip()
            l = l.split(",")
            l = [int(n) for n in l]
            cat_id, img = l[0], l[1:]
            img = np.asarray(img, dtype=np.uint8).reshape(28, 28)
            cat_ids.append(cat_id)
            imgs.append(img)
    
    for i, (cat_id, img) in enumerate(zip(cat_ids, imgs)):
        export_img_dir = os.path.join(export_dir, str(cat_id))
        export_img_p = os.path.join(export_img_dir, f"img{i}.png")

        os.makedirs(export_img_dir, exist_ok=True)
        cv2.imwrite(export_img_p, img)

def decode_rbf_vectors_from_imgs(
    img_dir: Union[str, os.PathLike], 
) -> np.ndarray:
    """
    Returns
    - `rbf_vectors`: `np.ndarray`, shape `(10, 84)`
    """
    filenames = range(10)
    filenames = [f"{i}_RBF.jpg" for i in filenames]

    rbf_vectors = []

    for filename in filenames:
        img_p = os.path.join(img_dir, filename)
        img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
        rbf_vector = np.where(img < 127, 1, -1).flatten()
        rbf_vectors.append(rbf_vector)
    
    rbf_vectors = np.stack(rbf_vectors, axis=0)

    return rbf_vectors
