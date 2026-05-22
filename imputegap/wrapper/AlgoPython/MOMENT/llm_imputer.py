'''

@author Maurice Amon
'''

from momentfm import MOMENTPipeline
from momentfm.utils.utils import control_randomness
from momentfm.data.informer_dataset import InformerDataset
from torch.utils.data import DataLoader
import torch

# Set a seed to guarantee reproducibility ...
control_randomness(seed=13)

# Load the model ...
model = MOMENTPipeline.from_pretrained(
    "AutonLab/MOMENT-1-large",
    model_kwargs={'task_name': 'reconstruction'} # For imputation, we will load MOMENT in `reconstruction` mode
    # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
)

model.init()
print(model)



test_dataset = InformerDataset(
    data_split='test',
    task_name='imputation',
    data_stride_len=512)

test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

import numpy as np
import matplotlib.pyplot as plt

n_channels = test_dataset[0][0].shape[0]
idx = np.random.randint(0, len(test_dataset))
channel_idx = np.random.randint(0, n_channels)
plt.plot(test_dataset[idx][0][channel_idx, :].squeeze(), c='darkblue')
plt.title(f'idx={idx}  | channel={channel_idx}')
plt.show()

from momentfm.utils.masking import Masking

mask_generator = Masking(mask_ratio=0.25) # Mask 25% of patches randomly

from tqdm import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = model.to(device).float()

trues, preds, masks = [], [], []
with torch.no_grad():
    for batch_x, batch_masks in tqdm(test_dataloader, total=len(test_dataloader)):
        trues.append(batch_x.numpy())

        batch_x = batch_x.to(device).float()
        n_channels = batch_x.shape[1]

        # Reshape to [batch_size * n_channels, 1, window_size]
        batch_x = batch_x.reshape((-1, 1, 512))

        batch_masks = batch_masks.to(device).long()
        batch_masks = batch_masks.repeat_interleave(n_channels, axis=0)

        mask = mask_generator.generate_mask(
            x=batch_x, input_mask=batch_masks).to(device).long()

        output = model(x_enc=batch_x, input_mask=batch_masks, mask=mask)  # [batch_size, n_channels, window_size]

        reconstruction = output.reconstruction.detach().cpu().numpy()
        mask = mask.detach().squeeze().cpu().numpy()

        # Reshape back to [batch_size, n_channels, window_size]
        reconstruction = reconstruction.reshape((-1, n_channels, 512))
        mask = mask.reshape((-1, n_channels, 512))

        preds.append(reconstruction)
        masks.append(mask)

preds = np.concatenate(preds)
trues = np.concatenate(trues)
masks = np.concatenate(masks)

print(f"Shapes: preds={preds.shape} | trues={trues.shape} | masks={masks.shape}")


from momentfm.utils.forecasting_metrics import mse, mae

print(f"Mean Squarred Error (MSE)={mse(y=trues[masks==0], y_hat=preds[masks==0], reduction='mean')}")
print(f"Mean Absolute Error (MAE)={mae(y=trues[masks==0], y_hat=preds[masks==0], reduction='mean')}")