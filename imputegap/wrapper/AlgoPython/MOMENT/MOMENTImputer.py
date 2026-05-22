'''

@author Maurice Amon
'''

import os
import sys

# Ensure local momentfm is used instead of site-packages
moment_dir = os.path.dirname(os.path.abspath(__file__))
if moment_dir not in sys.path:
    sys.path.insert(0, moment_dir)

from momentfm.utils.utils import control_randomness
from momentfm import MOMENTPipeline
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from momentfm.utils.masking import Masking

from imputegap.wrapper.AlgoPython.MOMENT.CustomDataset import CustomDataset

control_randomness(seed=13)


class MOMENTImputer:
    
    _model = None

    _test_dataset = None

    _test_dataloader = None

    _device = None

    def __init__(self, model):
        self._model = model


    def impute(self, batch_x, batch_masks):
        trues, preds, masks = [], [], []
        
        # Ensure we use the same device as the model
        device = next(self._model.parameters()).device
        
        with torch.no_grad():
            n_channels = batch_x.shape[1]
            trues.append(batch_x.cpu().numpy())
            batch_x = batch_x.to(device).float()
            
            # Reshaping to [batch size * n channels, window-size]
            window_size = batch_x.shape[2]
            batch_x = batch_x.reshape(-1, 1, window_size)
            batch_masks = batch_masks.to(device).long()
            batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

            # We don't want to generate a random mask for recovery,
            # we want to impute the ALREADY MISSING values.
            # In MOMENT, the 'mask' parameter specifies which patches to RECONSTRUCT.
            # A value of 1 in the mask means the patch is masked and should be reconstructed.
            # However, the input_mask already tells the model which values are observed.
            # For imputation, we usually want to reconstruct the whole sequence or specific parts.
            # If we don't provide a mask, it might default to something.
            # In the previous implementation, it used a random mask_ratio of 0.3.
            
            # Let's create a mask that covers the ENTIRE sequence, 
            # so the model attempts to reconstruct everything.
            # The input_mask will still guide it on what's actually there.
            
            mask = torch.ones((batch_x.shape[0], window_size), device=device).long()

            output = self._model(x_enc=batch_x, input_mask=batch_masks, mask=mask)
            reconstruction = output.reconstruction.detach().cpu().numpy()
            mask = mask.detach().cpu().numpy()

            # Reshape back to [batch size, n channels, window-size]
            reconstruction = reconstruction.reshape((-1, n_channels, window_size))
            mask = mask.reshape((-1, n_channels, window_size))
            preds.append(reconstruction)
            masks.append(mask)

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        masks = np.concatenate(masks)

        preds = preds.reshape(-1, preds.shape[-1])
        return preds



import numpy as np
import torch


def to_moment_imputation_format(
    data: np.ndarray,
    window_size: int = 512,
    stride: int = 512,
    channels_first: bool = True,
    fill_value: float = 0.0,
):
    if not channels_first:
        data = data.T  # (C, T)

    C, T = data.shape

    # valid timestep if NOT ALL channels are NaN
    input_mask_full = (~np.all(np.isnan(data), axis=0)).astype(np.int64)

    data_filled = np.nan_to_num(data, nan=fill_value).astype(np.float32)

    # If the timeseries is shorter than the window_size, pad it with zeros
    if T < window_size:
        pad_len = window_size - T
        data_filled = np.pad(data_filled, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)
        input_mask_full = np.pad(input_mask_full, (0, pad_len), mode='constant', constant_values=0)
        T = window_size

    x_windows, mask_windows = [], []
    for s in range(0, T - window_size + 1, stride):
        e = s + window_size
        x_windows.append(data_filled[:, s:e])       # (C, 512)
        mask_windows.append(input_mask_full[s:e])   # (512,)

    x_enc = torch.tensor(np.stack(x_windows), dtype=torch.float32)      # (N, C, 512)
    input_mask = torch.tensor(np.stack(mask_windows), dtype=torch.long) # (N, 512)

    return x_enc, input_mask


'''
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data = np.array([
        [5, 2, np.nan, 4, 1, 7],
        [3, np.nan, 6, 8, 2, 9],
    ], dtype=np.float32)

    data = np.tile(data, (1, 300))  # (2, 1800)


    x_enc, input_mask = to_moment_imputation_format(
        data,
        window_size=512,
        stride=512,
        channels_first=True
    )

    mask_generator = Masking(mask_ratio=0.3)


    print("x_enc:", x_enc.shape)
    print("input_mask:", input_mask.shape)

    print("Example input_mask sum (window 0):", input_mask[0].sum().item())

    x_enc = x_enc.to(device).float()
    input_mask = input_mask.to(device).long()
    momentimputer = MOMENTImputer()
    momentimputer.init_dataset()
    momentimputer.impute(x_enc, input_mask)'''