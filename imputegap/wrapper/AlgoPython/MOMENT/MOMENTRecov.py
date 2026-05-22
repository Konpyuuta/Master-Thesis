'''

@author Maurice Amon
'''

import numpy as np
import torch
import os
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm

from imputegap.recovery.manager import TimeSeries
from imputegap.wrapper.AlgoPython.MOMENT.MOMENTZeroshot import MOMENTZeroshot
from imputegap.wrapper.AlgoPython.MOMENT.finetuning.LPFinetuning_NP import LPFinetuning_NP
from imputegap.wrapper.AlgoPython.MOMENT.momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.masking import Masking

# Ensure local momentfm is used ...
moment_dir = os.path.dirname(os.path.abspath(__file__))
if moment_dir not in sys.path:
    sys.path.insert(0, moment_dir)



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


def _hidden_observed_patch_mask(observed_mask, artificial_mask, patch_len):
    eligible_patch_mask = Masking.convert_seq_to_patch_view(
        observed_mask,
        patch_len=patch_len,
    )
    eligible_mask = Masking.convert_patch_to_seq_view(
        eligible_patch_mask,
        patch_len=patch_len,
    )
    eligible_mask = eligible_mask[..., :observed_mask.shape[-1]]
    return eligible_mask * (1 - artificial_mask)


def evaluate_imputation(model, dataloader, device, mask_ratios=[0.25], return_metrics=False,
                        self_supervised_masking=False):
    model.eval()
    all_mse = []
    all_mae = []
    final_reconstruction = None

    with torch.no_grad():
        for mask_ratio in mask_ratios:
            mask_generator = Masking(mask_ratio=mask_ratio)
            mse_vals = []
            mae_vals = []
            reconstructions = []
            for batch_x, batch_masks in tqdm(dataloader, desc=f"Evaluating (mask={mask_ratio})", disable=False):
                batch_x = batch_x.to(device).float()
                n_channels = batch_x.shape[1]
                window_size = batch_x.shape[2]

                model_input = torch.nan_to_num(batch_x, nan=0.0)

                if self_supervised_masking:
                    batch_masks = _flatten_observed_mask(batch_masks, batch_x)
                else:
                    batch_masks = batch_masks.to(device).long()
                    batch_masks = batch_masks.repeat_interleave(n_channels, dim=0)

                model_input = model_input.reshape(-1, 1, window_size)
                mask = mask_generator.generate_mask(x=model_input, input_mask=batch_masks).to(device).long()
                input_mask = batch_masks
                if self_supervised_masking:
                    input_mask = torch.ones_like(batch_masks, device=device).long()

                output = model(x_enc=model_input, input_mask=input_mask, mask=mask)
                pred = output.reconstruction

                if self_supervised_masking:
                    eval_mask = _hidden_observed_patch_mask(
                        batch_masks,
                        mask,
                        patch_len=mask_generator.patch_len,
                    ).unsqueeze(1)
                else:
                    # Keep the original MOMENT evaluation convention for the paper-style
                    # original=True path. Changing this mask shifts the reported metrics.
                    eval_mask = (batch_masks * mask).unsqueeze(1)

                sq_err = ((pred - model_input) ** 2) * eval_mask
                abs_err = torch.abs(pred - model_input) * eval_mask

                mse = sq_err.nansum() / (eval_mask.nansum() + 1e-7)
                mae = abs_err.nansum() / (eval_mask.nansum() + 1e-7)

                mse_vals.append(mse.item())
                mae_vals.append(mae.item())

                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                reconstructions.append(pred.reshape(-1, n_channels, window_size).cpu().numpy())

            all_mse.append(np.mean(mse_vals))
            all_mae.append(np.mean(mae_vals))

            final_reconstruction = np.concatenate(reconstructions, axis=0)
            print(f"Mask Ratio {mask_ratio}: MSE={all_mse[-1]:.4f}, MAE={all_mae[-1]:.4f}")

    reconstructed_data = MOMENTZeroshot._windows_to_timeseries(
        final_reconstruction,
        dataset=getattr(dataloader, "dataset", None),
    )
    if return_metrics:
        return np.mean(all_mse), np.mean(all_mae), reconstructed_data
    return reconstructed_data


def evaluate_imputation_contaminated(model, dataloader, device):
    model.eval()
    mse_vals = []
    mae_vals = []
    reconstructions = []

    with torch.no_grad():
        for batch_x, batch_masks in tqdm(dataloader, desc="Evaluating (contaminated)", disable=False):
            batch_x = batch_x.to(device).float()
            batch_masks = batch_masks.to(device).long()
            n_channels = batch_x.shape[1]

            # [B, C, L] -> [B*C, 1, L]
            batch_x = batch_x.reshape(-1, 1, 512)
            batch_masks = batch_masks.repeat_interleave(n_channels, dim=0)
            original = batch_x.clone()

            # For already contaminated data, batch_masks indicates missing values (0)
            # MOMENT mask: 1 for patches to-be-reconstructed, 0 for observed.
            mask = 1 - batch_masks

            output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask)
            pred = output.reconstruction

            # For evaluation, we only look at the masked (contaminated) positions
            eval_mask = mask
            eval_mask = eval_mask.unsqueeze(1)

            sq_err = ((pred - original) ** 2) * eval_mask
            abs_err = torch.abs(pred - original) * eval_mask

            mse = sq_err.nansum() / (eval_mask.nansum() + 1e-7)
            mae = abs_err.nansum() / (eval_mask.nansum() + 1e-7)

            mse_vals.append(mse.item())
            mae_vals.append(mae.item())

            reconstructions.append(pred.reshape(-1, n_channels, 512).cpu().numpy())

    final_reconstruction = np.concatenate(reconstructions, axis=0)

    # Reshape final_reconstruction to (C, T)
    N, C, L = final_reconstruction.shape
    final_reconstruction = final_reconstruction.transpose(1, 0, 2).reshape(C, -1)

    return np.mean(mse_vals), np.mean(mae_vals), final_reconstruction

def _ensure_imputegap_shape(reconstructed_data, ts_data):
    if reconstructed_data.shape == ts_data.shape:
        return reconstructed_data
    if reconstructed_data.T.shape == ts_data.shape:
        return reconstructed_data.T

    raise ValueError(
        "MOMENT returned shape "
        f"{reconstructed_data.shape}, expected {ts_data.shape}."
    )

def _window_starts_for_target_windows(length, seq_len, target_windows):
    if target_windows is None or target_windows <= 0 or length <= seq_len:
        return None

    last_start = length - seq_len
    target_windows = min(target_windows, last_start + 1)
    return np.rint(np.linspace(0, last_start, target_windows)).astype(int)

def _window_starts_for_target_batches(length, seq_len, batch_size, target_batches):
    if target_batches is None or target_batches <= 0:
        return None

    target_windows = max(1, (target_batches - 1) * batch_size + 1)
    return _window_starts_for_target_windows(length, seq_len, target_windows)

def _auto_zeroshot_target_windows(ts_data, tr_ratio, scaler):
    full_series = ts_data.shape[0] < 14400
    reference_split = InformerDataset(data_split='test', task_name='imputation',
                                      data_stride_len=1,
                                      ts_m=ts_data, tr_ratio=tr_ratio,
                                      scaler=scaler, interpolate_missing=False,
                                      full_series=full_series)
    return len(reference_split)

def _imputation_dataset(ts_data, data_split, data_stride_len, tr_ratio, scaler=None,
                        interpolate_missing=True, window_starts=None, full_series=False,
                        return_observed_mask=False):
    return InformerDataset(data_split=data_split, task_name='imputation',
                           data_stride_len=data_stride_len,
                           ts_m=ts_data, tr_ratio=tr_ratio, scaler=scaler,
                           interpolate_missing=interpolate_missing,
                           window_starts=window_starts, full_series=full_series,
                           return_observed_mask=return_observed_mask)

def _imputation_loader(dataset, batch_size, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def _fit_fewshot_model(ts_m, model, ts_data, tr_ratio, batch_size, self_supervised_masking):
    train_dataset = _imputation_dataset(
        ts_data,
        'train',
        1,
        tr_ratio,
        interpolate_missing=not self_supervised_masking,
        return_observed_mask=self_supervised_masking,
    )
    train_loader = _imputation_loader(train_dataset, batch_size, shuffle=True)
    finetuner = LPFinetuning_NP(
        ts_obj=ts_m,
        model=model,
        self_supervised_masking=self_supervised_masking,
    )
    trained_model = finetuner.start_finetuning(train_loader)
    torch.cuda.empty_cache()
    return trained_model, train_dataset

def _recovery_loader(ts_data, train_dataset, data_stride_len, batch_size, tr_ratio,
                     self_supervised_masking):
    recovery_dataset = _imputation_dataset(
        ts_data, 'test', data_stride_len, tr_ratio,
        scaler=train_dataset.scaler,
        interpolate_missing=False,
        full_series=True,
        return_observed_mask=self_supervised_masking,
    )
    return _imputation_loader(recovery_dataset, batch_size, shuffle=False)

def _evaluate_original_fewshot(trained_model, ts_data, tr_ratio, batch_size, device,
                               self_supervised_masking):
    if self_supervised_masking:
        return

    test_dataset = _imputation_dataset(
        ts_data,
        'test',
        1,
        tr_ratio,
        interpolate_missing=not self_supervised_masking,
        return_observed_mask=self_supervised_masking,
    )

    test_loader = _imputation_loader(test_dataset, batch_size, shuffle=False)
    mse, mae, _ = evaluate_imputation(
        trained_model,
        test_loader,
        device,
        mask_ratios=[0.125, 0.25, 0.375, 0.5],
        return_metrics=True,
        self_supervised_masking=self_supervised_masking,
    )
    print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

def _fewshot_reconstruction(ts_m, model, ts_data, tr_ratio, batch_size, data_stride_len,
                            device, evaluate_original, self_supervised_masking):
    trained_model, train_dataset = _fit_fewshot_model(
        ts_m,
        model,
        ts_data,
        tr_ratio,
        batch_size,
        self_supervised_masking,
    )
    if evaluate_original and not self_supervised_masking:
        _evaluate_original_fewshot(
            trained_model,
            ts_data,
            tr_ratio,
            batch_size,
            device,
            self_supervised_masking,
        )

    description = "Self-supervised imputation" if self_supervised_masking else "Few-shot imputation"
    return _impute_with_model(
        model=trained_model,
        dataloader=_recovery_loader(
            ts_data,
            train_dataset,
            data_stride_len,
            batch_size,
            tr_ratio,
            self_supervised_masking,
        ),
        device=device,
        description=description,
    )

def _zeroshot_window_starts(ts_data, seq_len, batch_size, tr_ratio, scaler, zeroshot_batches):
    if isinstance(zeroshot_batches, str) and zeroshot_batches.lower() == "auto":
        target_windows = _auto_zeroshot_target_windows(
            ts_data=ts_data,
            tr_ratio=tr_ratio,
            scaler=scaler,
        )
        return _window_starts_for_target_windows(
            length=ts_data.shape[0],
            seq_len=seq_len,
            target_windows=target_windows,
        )

    return _window_starts_for_target_batches(
        length=ts_data.shape[0],
        seq_len=seq_len,
        batch_size=batch_size,
        target_batches=zeroshot_batches,
    )

def _zeroshot_reconstruction(ts_data, tr_ratio, seq_len, batch_size, data_stride_len,
                             zeroshot_batches, seed, device):
    reference_dataset = _imputation_dataset(ts_data, 'train', 1, tr_ratio)
    window_starts = _zeroshot_window_starts(
        ts_data=ts_data,
        seq_len=seq_len,
        batch_size=batch_size,
        tr_ratio=tr_ratio,
        scaler=reference_dataset.scaler,
        zeroshot_batches=zeroshot_batches,
    )
    test_dataset = _imputation_dataset(
        ts_data, 'test', data_stride_len, tr_ratio,
        scaler=reference_dataset.scaler,
        interpolate_missing=False,
        window_starts=window_starts,
        full_series=True,
    )
    test_loader = _imputation_loader(test_dataset, batch_size, shuffle=False)

    tester = MOMENTZeroshot(seed=seed)

    return _impute_with_model(
        model=tester.model,
        dataloader=test_loader,
        device=device,
        description="Zero-shot imputation",
    )

def _impute_with_model(model, dataloader, device, description):
    imputer = MOMENTZeroshot.__new__(MOMENTZeroshot)
    imputer.model = model.to(device).float()
    imputer.device = device
    reconstructed_data = imputer.run_test(
        test_dataloader=dataloader,
        description=description,
    )

    dataset = getattr(dataloader, "dataset", None)
    scaler = getattr(dataset, "scaler", None)
    if scaler is not None and hasattr(scaler, "mean_") and reconstructed_data.size:
        reconstructed_data = scaler.inverse_transform(reconstructed_data)

    return reconstructed_data

def recovMOMENT(ts_m, model="AutonLab/MOMENT-1-large", zeroshot=True, freq="h", seq_len=512, batch_size=64, data_stride_len=None, epochs=1, num_workers=0, seed=2023, tr_ratio=0.7, verbose=True, deep_verbose=False, normalization=False, scaling=True, shuffle=True, strat="seq", original=True, zeroshot_batches=None, self_supervised_masking=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if data_stride_len is None:
        data_stride_len = seq_len

    if isinstance(ts_m, TimeSeries):
        ts_data = ts_m.data  # This is (T, C) if reverse=False (default)
    else:
        ts_data = np.asarray(ts_m)

    percentage_masked = np.isnan(ts_data).mean() * 100
    print(f"Percentage of masked values: {percentage_masked:.2f}%")

    if zeroshot:
        reconstructed_data = _zeroshot_reconstruction(
            ts_data=ts_data,
            tr_ratio=tr_ratio,
            seq_len=seq_len,
            batch_size=batch_size,
            data_stride_len=data_stride_len,
            zeroshot_batches=zeroshot_batches,
            seed=seed,
            device=device,
        )
    else:
        reconstructed_data = _fewshot_reconstruction(
            ts_m=ts_m,
            model=model,
            ts_data=ts_data,
            tr_ratio=tr_ratio,
            batch_size=batch_size,
            data_stride_len=data_stride_len,
            device=device,
            evaluate_original=original and not self_supervised_masking,
            self_supervised_masking=self_supervised_masking,
        )

    return _ensure_imputegap_shape(reconstructed_data, ts_data)
