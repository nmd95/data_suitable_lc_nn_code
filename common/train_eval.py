import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy

from .data_utils import permute_elements


class TensorDataSet(Dataset):
    """A simple dataset class for loading tensor data."""

    def __init__(self, data_tensor: torch.Tensor, labels_tensor: torch.Tensor, device: str):
        """
        Initialize the dataset.

        Args:
            data_tensor (torch.Tensor): The data tensor.
            labels_tensor (torch.Tensor): The labels tensor.
            device (str): The device to load the data on.
        """
        self.data_tensor = data_tensor
        self.labels_tensor = labels_tensor
        self._device = device

    def __len__(self):
        """Get the length of the dataset."""
        return int(self.data_tensor.shape[0])

    def __getitem__(self, idx):
        """Get a single item from the dataset by index."""
        # Load the data and label for the given index
        data = self.data_tensor[idx, :].float().to(self._device)
        label = self.labels_tensor[idx].type(torch.LongTensor).to(self._device)

        return data, label


def _set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
            self,
            model: nn.Module,
            criterion,
            optimizer,
            scheduler,
            num_classes,
            permutation: List[int],
            device: str,
            data_dir: str,
            batch_sizes: List[int],
            data_dims=None,
            seed: int = None,  # Add 'seed' parameter
            weights_save_path: Optional[str] = None
    ) -> None:

        self.model = model
        self.best_model = model.state_dict()
        self.best_model_epoch = 0
        self.best_val_acc = -1.0
        self.best_model_test_acc = -1.0
        self.best_model_train_acc = -1.0
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.epoch_train_losses = []
        self.epoch_train_acc = []
        self.epoch_val_acc = []

        self.epoch_counter = 0

        self.permutation = permutation

        self.device = device

        self.weights_save_path = weights_save_path

        self.data_dims = data_dims

        self.seed = seed  # Save the seed value
        _set_seed(seed)  # Set the random seed

        self.data_dir = data_dir

        self.batch_sizes = batch_sizes

        self.train_dl = None
        self.val_dl = None
        self.test_dl = None

        self._on_init_callback()

    # Rest of the code remains the same

    def make_data_loaders(self):

        train_data = torch.from_numpy(np.load(f"{self.data_dir}/train_data.npy"))
        val_data = torch.from_numpy(np.load(f"{self.data_dir}/val_data.npy"))
        test_data = torch.from_numpy(np.load(f"{self.data_dir}/test_data.npy"))

        model_class = type(self.model)

        if model_class.__name__ != "S4Model":
            train_data = train_data.unsqueeze(1)
            val_data = val_data.unsqueeze(1)
            test_data = test_data.unsqueeze(1)

        train_labels = torch.from_numpy(np.load(f"{self.data_dir}/train_labels.npy"))
        val_labels = torch.from_numpy(np.load(f"{self.data_dir}/val_labels.npy"))
        test_labels = torch.from_numpy(np.load(f"{self.data_dir}/test_labels.npy"))

        train_ds = TensorDataSet(train_data, train_labels, self.device)
        val_ds = TensorDataSet(val_data, val_labels, self.device)
        test_ds = TensorDataSet(test_data, test_labels, self.device)

        train_dl = DataLoader(train_ds, batch_size=self.batch_sizes[0], shuffle=True, drop_last=False)
        val_dl = DataLoader(val_ds, batch_size=self.batch_sizes[1], shuffle=False, drop_last=False)
        test_dl = DataLoader(test_ds, batch_size=self.batch_sizes[2], shuffle=False, drop_last=False)

        return train_dl, val_dl, test_dl

    def run_epoch(self):
        epoch_acc = Accuracy(num_classes=self.num_classes, task="multiclass").to(self.device)
        self.model.train()
        epoch_total_loss = 0.0
        epoch_sample_counter = 0

        for x_batch, y_batch in self.train_dl:

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            if self.permutation is not None:
                x_batch = permute_elements(x_batch, self.permutation)

            self.optimizer.zero_grad()

            y_pred = self.model(x_batch)
            loss = self.criterion(y_pred, y_batch)
            loss.backward()
            self.optimizer.step()

            epoch_acc.update(y_pred, y_batch)
            epoch_total_loss += loss.item()
            epoch_sample_counter += x_batch.shape[0]

        self._on_epoch_callback(epoch_acc.compute().item(), epoch_total_loss / epoch_sample_counter)

    def _on_init_callback(self):
        print("Initializing")
        print("Making data loaders")
        self.train_dl, self.val_dl, self.test_dl = self.make_data_loaders()

    def _on_epoch_callback(self, *args):
        self.epoch_train_acc.append(args[0])
        self.epoch_train_losses.append(args[1])

        print(
            f"\n Finished train epoch number: {self.epoch_counter}, average loss: {self.epoch_train_losses[-1]}, average accuracy: {self.epoch_train_acc[-1]}"
        )
        if self.scheduler is not None:
            self.scheduler.step()

        self.validate()

        self.epoch_counter += 1

    def validate(self, model=None):
        val_model = self.model if model is None else model
        val_model.eval()

        val_acc = Accuracy(num_classes=self.num_classes, task="multiclass").to(self.device)

        for x_batch, y_batch in self.val_dl:

            if self.permutation is not None:
                x_batch = permute_elements(x_batch, self.permutation)

            y_pred = val_model(x_batch)

            val_acc.update(y_pred, y_batch)

        self._on_val_callback(val_acc.compute().item(), model)

    def _on_val_callback(self, *args):
        print(f"\n Finished validation, average val accuracy: {args[0]}")

        if args[1] is None:
            self.epoch_val_acc.append(args[0])
            if self.best_val_acc < self.epoch_val_acc[-1]:
                self.best_val_acc = self.epoch_val_acc[-1]
                self.best_model = self.model.state_dict()
                self.best_model_epoch = self.epoch_counter
    def test(self, model=None):
        temp_state_dict = self.model.state_dict()
        new_state_dict = self.best_model if model is None else model.state_dict()
        self.model.load_state_dict(new_state_dict)

        self.model.eval()

        test_acc = Accuracy(num_classes=self.num_classes, task="multiclass").to(self.device)

        for x_batch, y_batch in self.test_dl:

            if self.permutation is not None:
                x_batch = permute_elements(x_batch, self.permutation)

            y_pred = self.model(x_batch)

            test_acc.update(y_pred, y_batch)

        final_test_acc = test_acc.compute().item()
        self._on_test_callback(final_test_acc)
        self.model.load_state_dict(temp_state_dict)
        return final_test_acc

    def _on_test_callback(self, *args):
        print(f"\n Finished test, average test accuracy: {args[0]}")

    def _on_termination_callback(self):
        self.model.load_state_dict(self.best_model)

        self.best_model_test_acc = self.test(self.model)
        self.best_val_acc = self.epoch_val_acc[self.best_model_epoch]
        self.best_model_train_acc = self.epoch_train_acc[self.best_model_epoch]

        self.model = self.model.cpu()

        if self.weights_save_path is not None:
            torch.save(self.model.state_dict(), self.weights_save_path)

        print(
            f"\n Finished training for {self.epoch_counter} epochs. Best model obtained on epoch {self.best_model_epoch}, its val accuracy: {self.best_val_acc}, its test accuracy: {self.best_model_test_acc}."
        )

    def train_val_test(self, epochs):
        for _ in range(epochs):
            self.run_epoch()
        self._on_termination_callback()

    def get_final_results(self):
        return {
            "num_train_epochs": self.epoch_counter,
            "best_model_epoch": self.best_model_epoch,
            "best_val_acc": self.best_val_acc,
            "best_model_test_acc": self.best_model_test_acc,
            "epoch_train_losses": self.epoch_train_losses,
            "epoch_train_acc": self.epoch_train_acc,
            "epoch_val_acc": self.epoch_val_acc,
            "best_model_train_acc": self.best_model_train_acc,
            "weights_save_path": self.weights_save_path,
        }
