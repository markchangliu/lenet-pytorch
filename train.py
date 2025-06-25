import math
import multiprocessing as mp
import os
import shutil
from logging import Logger
from typing import Tuple, Union, Callable, Dict, Literal, Optional

import numpy as np
import torch
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from datasets import ImageFolderDataset
from models import LeNet5, loss_func
from loggers import setup_logging


def build_loader(
    train_data_root: Union[os.PathLike, str],
    test_data_root: Union[os.PathLike, str],
    train_transforms: Callable[[np.ndarray], torch.Tensor],
    test_transforms: Callable[[np.ndarray], torch.Tensor],
    cat_name_id_dict: Dict[str, int],
    batch_size: int, 
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    train_set = ImageFolderDataset(
        train_data_root, train_transforms, cat_name_id_dict
    )
    test_set = ImageFolderDataset(
        test_data_root, test_transforms, cat_name_id_dict
    )

    train_loader = DataLoader(
        train_set, batch_size, True, num_workers = num_workers, pin_memory = True
    )
    test_loader = DataLoader(
        test_set, batch_size, False, num_workers = num_workers, pin_memory = True
    )

    return train_loader, test_loader

@torch.no_grad()
def eval_model(
    model: LeNet5,
    test_loader: DataLoader,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eval_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Tuple[float, float]:
    """
    Returns
    - `loss_val`: `float`
    - `metric_val`: `float`
    """
    if model.training:
        model.eval() 

    batch_size = test_loader.batch_size
    curr_iters = 0
    cat_ids_all = []
    preds_all = []

    for batch_id, (imgs, cat_ids) in enumerate(test_loader):
        curr_iters += batch_size

        imgs: torch.Tensor
        cat_ids: torch.Tensor
        imgs = imgs.to("cuda:0", non_blocking = True)
        cat_ids = imgs.to("cuda:0", non_blocking = True)

        preds = model(imgs)
        loss = loss_func(cat_ids, preds)
        metric = eval_func(cat_ids, preds)

        assert loss.requires_grad is False and metric.requires_grad is False
    
        cat_ids_all.append(cat_ids)
        preds_all.append(preds)

    cat_ids_all = torch.concat(cat_ids_all)
    preds_all = torch.concat(preds_all)
    loss_all = loss_func(cat_ids_all, preds_all)
    metric_all = eval_func(cat_ids_all, preds_all)
    loss_all_val = loss_all.item()
    metric_all_val = metric_all.item()

    return loss_all_val, metric_all_val

@torch.no_grad()
def save_ckpt(
    save_p: Union[os.PathLike, str],
    model: LeNet5,
    optimizer: Optimizer,
    curr_epoch: int,
) -> None:
    save_dict = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "curr_epoch": curr_epoch
    }
    torch.save(save_dict, save_p)


def train_epochs(
    model: LeNet5,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    eval_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    logger: Logger,
    save_dir: Union[os.PathLike, str],
    num_epoches: int,
    print_iter_period: int,
    save_eval_epoch_period: int,
) -> None:
    model.train()
    os.makedirs(save_dir, exist_ok = True)

    train_set_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    curr_batches = 0
    curr_epoches = 0
    curr_iters = 0
    num_iters = math.ceil(train_set_size / batch_size) * num_epoches

    best_model_info = {
        "epoch": 0, 
        "test_loss": 0,
        "test_metric": 0,
    }

    for epoch_id in range(num_epoches):
        curr_epoches += 1

        for batch_id, (imgs, cat_ids) in enumerate(train_loader):
            curr_batches += 1
            curr_iters += batch_size

            imgs = imgs.to("cuda:0", non_blocking = True)
            cat_ids = cat_ids.to("cuda:0", non_blocking = True)
            preds = model(imgs)
            loss = loss_func(cat_ids, preds)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_val = loss.item()

            if curr_iters % print_iter_period == 0:
                msg = f"Train epoch {curr_epoches}/{num_epoches}, "
                msg += f"iter {curr_iters}/{num_iters}, "
                msg += f"loss {loss_val:.4f}. "
                logger.info(msg)
            
        if curr_epoches % save_eval_epoch_period == 0:
            model.eval()

            test_loss_val, test_metric_val = eval_model(
                model, test_loader, loss_func, eval_func,
            )

            msg = f"Test epoch {curr_epoches}/{num_epoches}, "
            msg += f"loss {test_loss_val:.4f}, metric {test_metric_val:.4f}. "
            logger.info(msg)

            save_p = os.path.join(save_dir, f"epoch{curr_epoches}.pth")
            logger.info(f"Saving ckpt at '{save_dir}'")
            save_ckpt(save_p, model, optimizer, curr_epoches)

            if test_metric_val > best_model_info["test_metric"]:
                best_model_info["epoch"] = curr_epoches
                best_model_info["test_loss"] = test_loss_val
                best_model_info["test_metric"] = test_metric_val
            
            model.train()
    
    best_epoch = best_model_info["epoch"]
    best_model_name = f"epoch{best_epoch}.pth"
    best_ckpt_p = os.path.join(save_dir, best_model_name)
    best_ckpt_save_p = os.path.join(save_dir, "best.pth")
    shutil.copy(best_ckpt_p, best_ckpt_save_p)
