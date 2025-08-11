
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset




# --- Local Imports ---
from .logging import PML_Logger


class PML_BasicCallback:
    """
    Abstract base class for creating callbacks in PyTorch.

    A callback is a set of functions to be applied at specific stages of the
    training process. You can use callbacks to gain insight into the model's
    internal states and statistics during training.

    To create a custom callback, inherit from this class and implement the
    methods corresponding to the events you want to intercept.
    """
    def __init__(self):
        # The 'state' dictionary will be injected by the Trainer and will contain
        # all relevant information (model, optimizer, losses, etc.)
        self.state = None

    def on_train_begin(self):
        """Called at the beginning of training."""
        pass

    def on_train_end(self):
        """Called at the end of training."""
        pass

    def on_epoch_begin(self):
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self):
        """Called at the end of each epoch."""
        pass

    def on_batch_begin(self):
        """Called at the beginning of each training batch."""
        pass

    def on_batch_end(self):
        """Called at the end of each training batch."""
        pass

    def on_validation_begin(self):
        """Called at the beginning of the validation phase."""
        pass

    def on_validation_end(self):
        """Called at the end of the validation phase."""
        pass




class PML_BasicDataset(Dataset):
    def __init__(self, input_external_data, output_external_data):
        self.input_data  = torch.FloatTensor(input_external_data)
        self.output_data = torch.FloatTensor(output_external_data)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_data = self.input_data[idx]
        output_data = self.output_data[idx]
        
        return input_data, output_data

