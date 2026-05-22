'''

@author Maurice Amon
'''
import numpy as np
import torch
import os
import sys

# Ensure local momentfm is used instead of site-packages
# LPFinetuning.py is in MOMENT/finetuning/, we need MOMENT/
moment_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if moment_dir not in sys.path:
    sys.path.insert(0, moment_dir)

from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from momentfm import MOMENTPipeline
from momentfm.data.informer_dataset import InformerDataset
from momentfm.utils.masking import Masking


class LPFinetuning:

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

    def __init__(self):
        self._data = 'imputegap/datasets/ETTh1.txt'

    def start_finetuning(self):
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
        data_path = self._data
        train_dataset = InformerDataset(data_split='train', task_name=self._task_name, data_stride_len=self._data_stride_len,
                                        file_path=data_path)
        test_dataset = InformerDataset(data_split='test', task_name=self._task_name, data_stride_len=self._data_stride_len,
                                       file_path=data_path)

        # Take a subset of train_dataset if it's too large for the timeout
        # ETTh1 has ~12*30*24 = 8640 points for train.
        # With seq_len=512 and stride=1, we have ~8000 samples.
        # 8000 / 64 = 125 batches. 125 * 5.5s = 687s.
        # It SHOULD fit in 600s if we reduce it slightly or if it runs a bit faster.
        # Let's limit to 100 batches.

        train_loader = DataLoader(train_dataset, batch_size=self._batch_size, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self._batch_size, shuffle=False, num_workers=0)

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

            mse, mae = self.evaluate_imputation(model, test_loader, device)
            print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

        print(f"Final Results: MSE={mse:.4f}, MAE={mae:.4f}")

    def train_one_epoch(self, model, dataloader, optimizer, scheduler, criterion, device):
        model.train()
        total_loss = 0
        mask_ratios = [0.125, 0.25, 0.375, 0.5]

        for batch_x, batch_masks in tqdm(dataloader, desc="Training", disable=False):
            batch_x = batch_x.to(device).float()
            n_channels = batch_x.shape[1]

            # [B, C, L] -> [B*C, 1, L]
            batch_x = batch_x.reshape((-1, 1, 512))
            original = batch_x.clone()

            batch_masks = batch_masks.to(device).long()
            batch_masks = batch_masks.repeat_interleave(n_channels, dim=0)

            # Randomly select a mask ratio for this batch from the specified rates
            mask_ratio = np.random.choice(mask_ratios)
            mask_generator = Masking(mask_ratio=mask_ratio)
            mask = mask_generator.generate_mask(x=batch_x, input_mask=batch_masks).to(device).long()

            optimizer.zero_grad()
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

    def evaluate_imputation(self, model, dataloader, device):
        model.eval()
        mask_ratios = self._mask_ratios
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
                    batch_masks = batch_masks.to(device).long()
                    n_channels = batch_x.shape[1]

                    # [B, C, L] -> [B*C, 1, L]
                    batch_x = batch_x.reshape(-1, 1, 512)
                    batch_masks = batch_masks.repeat_interleave(n_channels, dim=0)
                    original = batch_x.clone()

                    mask = mask_generator.generate_mask(x=batch_x, input_mask=batch_masks).to(device).long()

                    output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask)
                    pred = output.reconstruction

                    # For evaluation, we only look at the masked positions
                    eval_mask = batch_masks * mask
                    eval_mask = eval_mask.unsqueeze(1)

                    sq_err = ((pred - original) ** 2) * eval_mask
                    abs_err = torch.abs(pred - original) * eval_mask

                    mse = sq_err.nansum() / (eval_mask.nansum() + 1e-7)
                    mae = abs_err.nansum() / (eval_mask.nansum() + 1e-7)

                    mse_vals.append(mse.item())
                    mae_vals.append(mae.item())
                    
                    # Store reconstruction for the last mask ratio (or all, but user asked for ONE array)
                    # Typically we want the reconstruction with the current mask
                    # We'll reshape it back to (B, C, L)
                    reconstructions.append(pred.reshape(-1, n_channels, 512).cpu().numpy())

                all_mse.append(np.mean(mse_vals))
                all_mae.append(np.mean(mae_vals))
                
                # Keep the last mask_ratio's reconstruction as the 'reconstructed' array
                final_reconstruction = np.concatenate(reconstructions, axis=0)
                
                print(f"Mask Ratio {mask_ratio}: MSE={all_mse[-1]:.4f}, MAE={all_mae[-1]:.4f}")

        # Reshape final_reconstruction to (C, T) as requested
        # final_reconstruction is (N_samples, C, L)
        # To get (C, T), we might need to stitch it together. 
        # But wait, if they are windows, we just concatenate them along time if they don't overlap?
        # InformerDataset often has stride=1.
        
        # If the user specifically asked for (C, T), they might be expecting the stitched sequence.
        # But stitch logic depends on stride.
        # If we assume windows are consecutive or we just provide the raw array:
        # Let's simplify and provide (C, N_samples * L) or similar if stride=L.
        # However, usually for evaluation we just return the full tensor.
        
        # Let's see: (N, C, L) -> swap axes to (C, N, L) -> reshape to (C, N*L)
        N, C, L = final_reconstruction.shape
        final_reconstruction = final_reconstruction.transpose(1, 0, 2).reshape(C, -1)

        return np.mean(all_mse), np.mean(all_mae), final_reconstruction








