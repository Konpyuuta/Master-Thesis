
'''

@author Maurice Amon
'''
import numpy as np
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local momentfm is used instead of site-packages
moment_dir = os.path.dirname(os.path.abspath(__file__))
if moment_dir not in sys.path:
    sys.path.insert(0, moment_dir)

from momentfm import MOMENTPipeline
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.utils import control_randomness

class MOMENTZeroshot:

    def __init__(self, seed=13, model_name="AutonLab/MOMENT-1-large"):
        self.seed = seed
        control_randomness(seed=self.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = MOMENTPipeline.from_pretrained(
            model_name,
            model_kwargs={'task_name': 'reconstruction'}
        )
        self.model.init()
        self.model = self.model.to(self.device).float()
        self.model.eval()

    @staticmethod
    def _expand_observed_mask(batch_masks, batch_x):
        if batch_masks is None:
            return torch.ones_like(batch_x, dtype=torch.long)

        batch_masks = batch_masks.to(batch_x.device).long()
        if batch_masks.ndim == 2:
            batch_masks = batch_masks.unsqueeze(1).expand(-1, batch_x.shape[1], -1)
        elif batch_masks.ndim != 3:
            raise ValueError(
                "MOMENTZeroshot expects batch masks with shape [B, L] or [B, C, L]."
            )

        return batch_masks

    @staticmethod
    def _windows_to_timeseries(windows, dataset=None):
        if windows.size == 0:
            return windows

        n_windows, n_channels, window_size = windows.shape
        stride = getattr(dataset, "data_stride_len", window_size)
        length = getattr(dataset, "length_timeseries", n_windows * window_size)
        window_starts = getattr(dataset, "window_starts", None)

        sums = np.zeros((n_channels, length), dtype=windows.dtype)
        counts = np.zeros((n_channels, length), dtype=np.float32)
        last_start = max(0, length - window_size)

        for window_idx, window in enumerate(windows):
            if window_starts is not None:
                start = min(int(window_starts[window_idx]), last_start)
            else:
                start = min(window_idx * stride, last_start)
            end = min(start + window_size, length)
            usable = end - start
            if usable <= 0:
                continue

            sums[:, start:end] += window[:, :usable]
            counts[:, start:end] += 1

        counts[counts == 0] = 1
        return (sums / counts).T

    def run_test(self, test_dataloader=None, description="Zero-shot imputation"):
        # Load dataset if not provided
        if test_dataloader is None:
            test_dataset = InformerDataset(
                data_split='test',
                task_name='imputation',
                file_path='data/ETTh1.csv',
                data_stride_len=512)

            test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        dataset = getattr(test_dataloader, "dataset", None)
        n_windows = len(dataset) if dataset is not None else "unknown"
        print(
            f"{description} on provided contaminated time series "
            f"({n_windows} windows, {len(test_dataloader)} batches)"
        )

        self.model.eval()
        imputed_windows = []

        with torch.no_grad():
            for batch_x, batch_masks in tqdm(test_dataloader, total=len(test_dataloader)):
                batch_x = batch_x.to(self.device).float()
                n_channels = batch_x.shape[1]
                window_size = batch_x.shape[2]

                provided_observed_mask = self._expand_observed_mask(batch_masks, batch_x)
                nan_observed_mask = (~torch.isnan(batch_x)).long()
                observed_mask = provided_observed_mask * nan_observed_mask

                model_input = torch.nan_to_num(batch_x, nan=0.0)

                # MOMENT receives each channel as a separate univariate series.
                model_input = model_input.reshape((-1, 1, window_size))
                observed_mask = observed_mask.reshape((-1, window_size))

                # input_mask is an attention mask. Existing missing values are represented
                # by mask tokens and should still be attendable during reconstruction.
                input_mask = torch.ones_like(observed_mask, device=self.device).long()

                output = self.model(
                    x_enc=model_input,
                    input_mask=input_mask,
                    mask=observed_mask,
                )

                imputed = torch.where(
                    observed_mask.unsqueeze(1).bool(),
                    model_input,
                    output.reconstruction,
                )
                imputed = torch.nan_to_num(imputed, nan=0.0, posinf=0.0, neginf=0.0)
                imputed = imputed.reshape((-1, n_channels, window_size))
                imputed_windows.append(imputed.detach().cpu().numpy())

        if not imputed_windows:
            return np.empty((0, 0))

        imputed_windows = np.concatenate(imputed_windows, axis=0)
        return self._windows_to_timeseries(
            imputed_windows,
            dataset=getattr(test_dataloader, "dataset", None),
        )
