import math
import multiprocessing as mp
import os
import shutil
from logging import Logger
from typing import Tuple, Union, Callable, Dict, Literal, Optional, List, Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer, SGD
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from datasets import ImageFolderDataset
from models import LeNet5, loss_func, eval_func
from utils import decode_rbf_vectors_from_imgs
from loggers import setup_logging


def build_loader(
    train_data_roots: List[Union[os.PathLike, str]],
    test_data_roots: List[Union[os.PathLike, str]],
    train_transforms: Callable[[np.ndarray], torch.Tensor],
    test_transforms: Callable[[np.ndarray], torch.Tensor],
    cat_name_id_dict: Dict[str, int],
    batch_size: int, 
    num_workers: int
) -> Tuple[DataLoader, DataLoader]:
    train_set = ImageFolderDataset(
        train_data_roots, train_transforms, cat_name_id_dict
    )
    test_set = ImageFolderDataset(
        test_data_roots, test_transforms, cat_name_id_dict
    )

    train_loader = DataLoader(
        train_set, batch_size, True, num_workers = num_workers, pin_memory = True
    )
    test_loader = DataLoader(
        test_set, batch_size, False, num_workers = num_workers, pin_memory = True
    )

    return train_loader, test_loader

def build_optimizer(
    model_params: Iterable[Union[nn.Parameter, Dict[str, nn.Parameter]]],
    lr_global: float,
    **other_params_global: float
) -> Optimizer:
    """
    example
    ```
    # gloabl
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam([var1, var2], lr=0.0001)

    # per layer
    optim.SGD([
            {'params': model.base.parameters(), 'lr': 1e-2},
            {'params': model.classifier.parameters()}
        ], lr=1e-3, momentum=0.9)
    ```
    """
    sgd = SGD(model_params, lr_global, **other_params_global)
    return sgd


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
        cat_ids = cat_ids.to("cuda:0", non_blocking = True)

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
    save_ckpt_dir: Union[os.PathLike, str],
    num_epoches: int,
    print_iter_period: int,
    save_eval_epoch_period: int,
) -> None:
    model.train()
    os.makedirs(save_ckpt_dir, exist_ok = True)

    train_set_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    curr_batches = 0
    curr_epoches = 0
    curr_iters = 0
    num_iters = train_set_size * num_epoches

    curr_print_times = 0
    curr_save_eval_times = 0

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

            if curr_iters // print_iter_period > curr_print_times:
                msg = f"Train epoch {curr_epoches}/{num_epoches}, "
                msg += f"iter {curr_iters}/{num_iters}, "
                msg += f"loss {loss_val:.4f}. "
                logger.info(msg)
                curr_print_times += 1
            
        if curr_epoches // save_eval_epoch_period > curr_save_eval_times:
            model.eval()

            test_loss_val, test_metric_val = eval_model(
                model, test_loader, loss_func, eval_func,
            )

            msg = f"Test epoch {curr_epoches}/{num_epoches}, "
            msg += f"loss {test_loss_val:.4f}, metric {test_metric_val:.4f}. "
            logger.info(msg)

            save_p = os.path.join(save_ckpt_dir, f"epoch{curr_epoches}.pth")
            logger.info(f"Saving ckpt at '{save_ckpt_dir}'")
            save_ckpt(save_p, model, optimizer, curr_epoches)

            if test_metric_val > best_model_info["test_metric"]:
                best_model_info["epoch"] = curr_epoches
                best_model_info["test_loss"] = test_loss_val
                best_model_info["test_metric"] = test_metric_val
            
            model.train()

            curr_save_eval_times += 1
    
    best_epoch = best_model_info["epoch"]
    best_model_name = f"epoch{best_epoch}.pth"
    best_ckpt_p = os.path.join(save_ckpt_dir, best_model_name)
    best_ckpt_save_p = os.path.join(save_ckpt_dir, "best.pth")
    shutil.copy(best_ckpt_p, best_ckpt_save_p)


def main() -> None:
    train_loader, test_loader = build_loader(
        ["/data/cliu/large_files/projects/minist/data/train"],
        ["/data/cliu/large_files/projects/minist/data/test"],
        ToTensor(),
        ToTensor(),
        {str(i): i for i in range(10)},
        4096,
        mp.cpu_count() // 2
    )

    rbf_vectors = decode_rbf_vectors_from_imgs(
        "/data/cliu/large_files/projects/minist/data/RBF_kernel"
    )
    model = LeNet5(torch.as_tensor(rbf_vectors)).to("cuda:0")
    optimizer = build_optimizer(
        model.parameters(), 1e-5, momentum = 0.9, weight_decay = 1e-4
    )
    export_dir = "/data/cliu/large_files/projects/minist/runtimes/train_v01"
    log_file = os.path.join(export_dir, "train_log.txt")
    save_ckpt_dir = os.path.join(export_dir, "ckpts")
    logger = setup_logging(log_file)

    train_epochs(
        model, 
        train_loader,
        test_loader,
        optimizer,
        loss_func,
        eval_func,
        logger,
        save_ckpt_dir,
        120,
        4096*4,
        4
    )


if __name__ == "__main__":
    main()