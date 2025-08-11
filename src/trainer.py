# --- Standard Library Imports ---
import random
from typing import List, Optional

# --- Third-Party Imports ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import sys   # para StreamHandler apontar ao stdout
from datetime import datetime
import time


# --- Local Imports ---
from .logging import PML_Logger
from .templates import PML_BasicCallback





class PML_ModelTrainer:
    """
    A comprehensive PyTorch model trainer class.

    Encapsulates data loading, epoch management, model checkpointing,
    callback hooks, prediction and visualization.

    Attributes:
        model (nn.Module): The PyTorch model to be trained.
        loss_fn: The loss function.
        optimizer: The optimization algorithm.
        device (str): The device ('cuda' or 'cpu') for computation.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn,
                 optimizer,
                 train_loader: DataLoader,
                 valid_loader: Optional[DataLoader] = None,
                 logger=None,
                 seed: Optional[int] = None,
                 b_deterministic: bool = True,
                 callbacks=None):
        # — Logging —
        if logger is None:
            self._logger_wrapper = PML_Logger(name_prefix="PML_ModelTrainer")
            self.logger = self._logger_wrapper.get_logger()
            self.logger.debug("No external logger provided; using default.")
            print("No logger provided. Using default PML_Logger.")
        else:
            if not isinstance(logger, PML_Logger):
                raise TypeError("Logger must be an instance of PML_Logger.")
            self._logger_wrapper = logger
            self.logger = logger.get_logger()

        # — Core components —
        self.model     = model
        self.loss_fn   = loss_fn
        self.optimizer = optimizer

        # — Device —
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.set_model_device(self.device)

        # — Seed —
        if seed is not None:
            self.set_seed(seed, b_deterministic=b_deterministic)
        else:
            self.set_seed(b_deterministic=b_deterministic)

        # — DataLoaders —
        self.set_loaders(train_loader, valid_loader)

        # — History & state —
        self.train_losses: List[float] = []
        self.valid_losses: List[float] = []
        self.total_epochs: int        = 0

        self.state: dict = {
            'model':        self.model,
            'optimizer':    self.optimizer,
            'loss_fn':      self.loss_fn,
            'epoch':        0,
            'batch_idx':    0,
            'train_loss':   0.0,
            'valid_loss':   0.0,
            'stop_training': False
        }

        # — Callbacks —
        self.callbacks: List[PML_BasicCallback] = list(callbacks) if callbacks else []
        for cb in self.callbacks:
            cb.state = self.state

        # — Internal step fns —
        self._train_step = self._make_train_step()
        self._val_step   = self._make_val_step()

        self.logger.info("PML_ModelTrainer initialized on device: %s", self.device)


    # — Callback dispatcher —
    def _call_callbacks(self, event_name: str):
        for cb in self.callbacks:
            getattr(cb, event_name)()


    # — Configuration —
    def set_loaders(self,
                    train_loader: DataLoader,
                    valid_loader: Optional[DataLoader] = None) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.logger.info("Loaders set (train + valid)")


    # def set_model_device(self, device: str) -> None:
    #     if device not in ('cuda', 'cpu'):
    #         self.logger.error("Invalid device '%s'. Choose 'cuda' or 'cpu'.", device)
    #         raise ValueError(f"Invalid device '{device}'")
    #     try:
    #         self.device = device
    #         self.model.to(self.device)
    #     except RuntimeError:
    #         fallback = 'cuda' if torch.cuda.is_available() else 'cpu'
    #         self.device = fallback
    #         self.model.to(fallback)
    #         self.logger.warning("Falling back to device '%s'", fallback)
    #     self.logger.info("Model moved to %s", self.device)


    def set_model_device(self, device: str) -> None:
        if device not in ('cuda', 'cpu'):
            self.logger.error("Invalid device '%s'. Choose 'cuda' or 'cpu'.", device)
            raise ValueError(f"Invalid device '{device}'")
        try:
            self.device = device
            if device == 'cuda' and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                self.logger.info("Using %d GPUs with DataParallel.", torch.cuda.device_count())
            self.model.to(self.device)
        except RuntimeError:
            fallback = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.device = fallback
            if fallback == 'cuda' and torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
                self.logger.info("Using %d GPUs with DataParallel.", torch.cuda.device_count())
            self.model.to(fallback)
            self.logger.warning("Falling back to device '%s'", fallback)
        self.logger.info("Model moved to %s", self.device)


    def set_seed(self,
                 seed: Optional[int] = 42,
                 *,
                 b_deterministic: bool = True) -> int:
        if not b_deterministic:
            seed = time.time_ns() & 0xFFFF_FFFF
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark     = not b_deterministic
        torch.backends.cudnn.deterministic = b_deterministic
        # try:
        #     torch.use_deterministic_algorithms(b_deterministic)
        # except AttributeError:
        #     pass
        self.logger.info("Seed set to %d (deterministic=%s)", seed, b_deterministic)
        return seed


    # — Training + validation loop —
    def fit(self,
            n_epochs: int,
            *,
            n_starting_epoch: int = 0,
            best_model_path: str = "best_model.pth") -> dict:
        self.state['stop_training'] = False
        self.state['epoch']         = n_starting_epoch
        best_valid_loss = float('inf')

        self._call_callbacks('on_train_begin')

        all_losses = {"train": [], "valid": []}

        for _ in range(n_epochs):
            # epoch begin
            self.state['epoch'] += 1
            current = self.state['epoch']
            self._call_callbacks('on_epoch_begin')

            # — train —
            train_loss = self._run_epoch(validation=False)
            self.train_losses.append(train_loss)
            self.state['train_loss'] = train_loss
            all_losses["train"].append(train_loss)

            # — valid —
            valid_loss = None
            if self.valid_loader:
                self._call_callbacks('on_validation_begin')
                with torch.no_grad():
                    valid_loss = self._run_epoch(validation=True)
                self.valid_losses.append(valid_loss)
                self.state['valid_loss'] = valid_loss
                all_losses["valid"].append(valid_loss)
                self._call_callbacks('on_validation_end')

                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    model_to_save = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                    torch.save(model_to_save.state_dict(), best_model_path)
                    print(f"[Epoch {current}] Saved new best (valid_loss={valid_loss:.4e})")

            # logging
            msg = f"[Epoch {current}/{n_epochs}] train_loss={train_loss:.4e}"
            if valid_loss is not None:
                msg += f", valid_loss={valid_loss:.4e}"
            self.logger.info(msg)
            print(msg)

            self._call_callbacks('on_epoch_end')

            if self.state['stop_training']:
                self.logger.warning("Early stopping triggered.")
                break

        self._call_callbacks('on_train_end')
        return all_losses


    def _run_epoch(self, validation: bool = False) -> float:
        loader  = self.valid_loader if validation else self.train_loader
        step_fn = self._val_step      if validation else self._train_step

        if loader is None:
            raise ValueError("Validation loader not set." if validation
                             else "Training loader not set.")

        epoch_losses: List[float] = []
        self.state['batch_idx']   = 0

        for x_batch, y_batch in loader:

            self.state['batch_idx'] += 1
            self._call_callbacks('on_batch_begin')

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            loss = step_fn(x_batch, y_batch)
            epoch_losses.append(loss)

            key = 'valid_loss' if validation else 'train_loss'
            self.state[key] = loss

            self._call_callbacks('on_batch_end')

        return float(np.mean(epoch_losses))


    # — Step factories —
    def _make_train_step(self):
        def perform_train_step(x: torch.Tensor, y: torch.Tensor) -> float:
            self.optimizer.zero_grad()
            self.model.train()
            y_hat = self.model(x)
            loss  = self.loss_fn(y_hat, y)
            loss.backward()
            self.optimizer.step()
            return loss.item()
        return perform_train_step

    def _make_val_step(self):
        def perform_valid_step(x: torch.Tensor, y: torch.Tensor) -> float:
            self.model.eval()
            y_hat = self.model(x)
            loss  = self.loss_fn(y_hat, y)
            return loss.item()
        return perform_valid_step


    # — Prediction —
    def predict_with_numpy(self, x_input: np.ndarray) -> np.ndarray:
        self.model.eval()
        x_tensor = torch.as_tensor(x_input, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            y_hat = self.model(x_tensor)
        return y_hat.detach().cpu().numpy()
