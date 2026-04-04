'''

@author Maurice Amon
'''

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

    def __init__(self):
        self._model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={'task_name': 'reconstruction'})
        self._model.init()

    def init_dataset(self):
        # Receives a tensor of shape [batch size, n channels, context length]
        x = torch.randn(1, 2, 256)
        output = self._model(x_enc=x)

        self._test_dataset = CustomDataset(
            data_split='test',
            task_name='imputation',
            data_stride_len=512
        )

        #print(len(self._test_dataset))
        self._test_dataloader = DataLoader(self._test_dataset, batch_size=1, shuffle=False)

        #print(self._test_dataset[0][0])
        n_channels = self._test_dataset[0][0].shape[0]

        idx = np.random.randint(0, len(self._test_dataset))
        channel_idx = np.random.randint(0, n_channels)
        '''
        plt.plot(self._test_dataset[idx][0][channel_idx, :].squeeze(), c='darkblue')
        plt.title(f'idx={idx}  | channel={channel_idx}')
        plt.show()'''

        mask_generator = Masking(mask_ratio=0.25)

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).float()

    def impute(self, batch_x, batch_masks):
        trues, preds, masks = [], [], []
        mask_generator = Masking(mask_ratio=0.3)

        with torch.no_grad():
            '''for batch_x, batch_masks in tqdm(self._test_dataloader, total=len(self._test_dataloader)):

                print(batch_x.numpy().shape)
                trues.append(batch_x.numpy())
                batch_x = batch_x.to(self._device).float()
                # Reshaping to [batch size * n channels, window-size]
                batch_x = batch_x.reshape(-1, 1, 512)
                batch_masks = batch_masks.to(self._device).long()
                batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

                mask = mask_generator.generate_mask(
                    x=batch_x, input_mask=batch_masks
                ).to(self._device).long()'''
            n_channels = batch_x.shape[1]
            print(batch_x.cpu().numpy().shape)
            trues.append(batch_x.cpu().numpy())
            batch_x = batch_x.to(self._device).float()
            # Reshaping to [batch size * n channels, window-size]
            window_size = batch_x.shape[2]
            batch_x = batch_x.reshape(-1, 1, window_size)
            batch_masks = batch_masks.to(self._device).long()
            batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)
            print("batch_x:", batch_x.shape)
            print("batch_masks:", batch_masks.shape)

            print("batch_masks unique:", torch.unique(batch_masks))
            print("batch_masks sum:", batch_masks.sum().item(), "/", batch_masks.numel())
            print("batch_masks mean:", batch_masks.float().mean().item())
            mask = mask_generator.generate_mask(
                x=batch_x, input_mask=batch_masks
            ).to(self._device).long()
            print(mask)
            print("Mask: \n", mask.shape)
            output = self._model(x_enc=batch_x, input_mask=batch_masks, mask=mask)
            reconstruction = output.reconstruction.detach().cpu().numpy()
            mask = mask.detach().squeeze().cpu().numpy()

            # Reshape back to [batch size, n channels, window-size]
            reconstruction = reconstruction.reshape((-1, n_channels, window_size))
            print(reconstruction.shape)
            print(mask.shape)
            print(n_channels)
            mask = mask.reshape((-1, n_channels, window_size))
            preds.append(reconstruction)
            masks.append(mask)

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        masks = np.concatenate(masks)

        print(f"Shapes: preds={preds.shape} | trues={trues.shape} | masks={masks.shape}")
        print(preds)
        print(trues)
        print(masks)
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