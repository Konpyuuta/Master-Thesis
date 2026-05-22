'''

@author Maurice Amon
'''
import numpy as np
import torch
import os
import sys

# Ensure local momentfm is used instead of site-packages
moment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if moment_dir not in sys.path:
    sys.path.insert(0, moment_dir)

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from momentfm import MOMENTPipeline
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.masking import Masking


class LPFinetuning_NP:

    # All hyperparameters that have been specified in the paper ..

    _data = None

    _model = "AutonLab/MOMENT-1-large"

    _seq_len = 512

    _patch_len = 8

    _patch_stride_len = 8

    _batch_size = 64

    _epochs = 1

    _learning_rate = 1e-4

    _max_lr = 1e-3

    _random_seed = 13

    _task_name = 'imputation'

    _data_stride_len = 1

    _mask_ratios = [0.125, 0.25, 0.375, 0.5]

    def __init__(self, ts_obj=None, model="AutonLab/MOMENT-1-large", seq_len=512, patch_len=8, patch_stride_len=8, batch_size=64, epochs=1, learning_rate=1e-4, max_lr=1e-3, random_seed=13, task_name='imputation', data_stride_len=1, self_supervised_masking=False):
        self._ts_obj = ts_obj
        self._model = model
        self._seq_len = seq_len
        self._patch_len = patch_len
        self._patch_stride_len = patch_stride_len
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._max_lr = max_lr
        self._random_seed = random_seed
        self._task_name = task_name
        self._data_stride_len = data_stride_len
        self._self_supervised_masking = self_supervised_masking

    def start_finetuning(self, train_loader):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load model
        model = MOMENTPipeline.from_pretrained(
            self._model,
            model_kwargs={
                'task_name': 'reconstruction',
                'seq_len': self._seq_len,
                'patch_len': self._patch_len,
                'patch_stride_len': self._patch_stride_len,
                'freeze_encoder': True,
                'freeze_embedder': True,
            }
        )
        model.init()
        model = model.to(device)

        # Verify linear probing (only head should have requires_grad=True)
        for name, param in model.named_parameters():
            if "head" in name:
                param.requires_grad = True
                print(f"Training parameter: {name}")
            else:
                param.requires_grad = False

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

        # Datasets
        # We need to adapt InformerDataset to accept numpy array if we want to use TimeSeries.data
        # However, the user said "just replace the .txt file loading as dataset and make it compatible with the libraries TimeSeries() loading"
        # If I want to use TimeSeries, I can pass the data to InformerDataset.
        
        # Let's check if we can pass the data directly.
        # Based on previous session history, InformerDataset was supposed to handle numpy arrays.
        # But the current informer_dataset.py doesn't show it.
        # I will modify InformerDataset to support a new 'ts_m' parameter.


        if len(train_loader) == 0:
            print("Error: Train loader is empty. Dataset might be too small for the given sequence length.")
            return

        # Training setup
        optimizer = torch.optim.Adam(trainable_params, lr=self._learning_rate)
        criterion = torch.nn.MSELoss(reduction="none")

        epochs = self._epochs
        scheduler = OneCycleLR(optimizer, max_lr=self._max_lr, steps_per_epoch=len(train_loader), epochs=epochs,
                               pct_start=0.3)

        print("Starting finetuning (Linear Probing)...")
        for epoch in range(epochs):
            avg_train_loss = self.train_one_epoch(model, train_loader, optimizer, scheduler, criterion, device)
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")


        return model

    @staticmethod
    def _flatten_observed_mask(batch_masks, batch_x):
        batch_masks = batch_masks.to(batch_x.device).long()
        if batch_masks.ndim == 2:
            batch_masks = batch_masks.unsqueeze(1).expand(-1, batch_x.shape[1], -1)
        elif batch_masks.ndim == 3:
            if batch_masks.shape[1] == 1 and batch_x.shape[1] != 1:
                batch_masks = batch_masks.expand(-1, batch_x.shape[1], -1)
            elif batch_masks.shape[1] != batch_x.shape[1]:
                raise ValueError(
                    "MOMENT observed masks must have shape [B, L] or [B, C, L]."
                )
        else:
            raise ValueError(
                "MOMENT observed masks must have shape [B, L] or [B, C, L]."
            )

        batch_masks = batch_masks * (~torch.isnan(batch_x)).long()
        return batch_masks.reshape(-1, batch_x.shape[2])

    def _hidden_observed_patch_mask(self, observed_mask, artificial_mask):
        eligible_patch_mask = Masking.convert_seq_to_patch_view(
            observed_mask,
            patch_len=self._patch_len,
            stride=self._patch_stride_len,
        )
        eligible_mask = Masking.convert_patch_to_seq_view(
            eligible_patch_mask,
            patch_len=self._patch_len,
        )
        eligible_mask = eligible_mask[..., :observed_mask.shape[-1]]
        return eligible_mask * (1 - artificial_mask)

    def train_one_epoch(self, model, dataloader, optimizer, scheduler, criterion, device):
        model.train()
        total_loss = 0
        mask_ratios = [0.125, 0.25, 0.375, 0.5]

        for batch_x, batch_masks in tqdm(dataloader, desc="Training", disable=False):
            batch_x = batch_x.to(device).float()
            n_channels = batch_x.shape[1]

            # Randomly select a mask ratio for this batch from the specified rates
            mask_ratio = np.random.choice(mask_ratios)
            mask_generator = Masking(mask_ratio=mask_ratio)

            optimizer.zero_grad()
            if self._self_supervised_masking:
                window_size = batch_x.shape[2]
                observed_mask = self._flatten_observed_mask(batch_masks, batch_x)

                # [B, C, L] -> [B*C, 1, L]
                batch_x = batch_x.reshape((-1, 1, window_size))
                original = batch_x.clone()
                model_input = torch.nan_to_num(batch_x, nan=0.0, posinf=0.0, neginf=0.0)

                mask = mask_generator.generate_mask(
                    x=model_input,
                    input_mask=observed_mask,
                ).to(device).long()
                input_mask = torch.ones_like(observed_mask, device=device).long()
                output = model(x_enc=model_input, input_mask=input_mask, mask=mask)

                target = torch.nan_to_num(original, nan=0.0, posinf=0.0, neginf=0.0)
                recon_loss = criterion(output.reconstruction, target)
                loss_mask = self._hidden_observed_patch_mask(observed_mask, mask).unsqueeze(1)
                loss = (loss_mask * recon_loss).sum() / (loss_mask.sum() + 1e-7)
            else:
                # [B, C, L] -> [B*C, 1, L]
                batch_x = batch_x.reshape((-1, 1, 512))
                original = batch_x.clone()

                batch_masks = batch_masks.to(device).long()
                batch_masks = batch_masks.repeat_interleave(n_channels, dim=0)
                mask = mask_generator.generate_mask(x=batch_x, input_mask=batch_masks).to(device).long()

                output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask)

                recon_loss = criterion(output.reconstruction, original)
                observed_mask = batch_masks * (1 - mask)
                masked_loss = observed_mask * recon_loss

                loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)

            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


