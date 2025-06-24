import os
from typing import Union, Callable, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ImageFolderDataset(Dataset):
    def __init__(
        self,
        data_roots: List[Union[str, os.PathLike]],
        transforms: Optional[Callable[[np.ndarray], torch.Tensor]],
        cat_name_id_dict: Dict[str, int]
    ) -> None:
        assert isinstance(data_roots, list)

        self.cat_name_id_dict = cat_name_id_dict
        self.cat_id_name_dict = {v: k for k, v in cat_name_id_dict.items()}
        self.transforms = transforms
        self.data = [] # (img_p, cat_id)

        dirnames = list(cat_name_id_dict.keys())
        dirnames.sort()

        for root in data_roots:
            for dirname in dirnames:
                img_dir = os.path.join(root, dirname)
                filenames = os.listdir(img_dir)
                filenames.sort()

                for filename in filenames:
                    if not filename.endswith((".png", ".jpg", ".jpeg")):
                        continue
                    
                    img_p = os.path.join(img_dir, filename)
                    cat_id = self.cat_name_id_dict[dirname]
                    self.data.append((img_p, cat_id))
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_p, cat_id = self.data[idx]
        img = cv2.imread(img_p)
        img_input = self.transforms(img) if self.transforms is not None else img
        return img_input, cat_id
