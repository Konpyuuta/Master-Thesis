import os

import numpy as np
import torch
import random
from datetime import datetime
import pathlib

from wrapper.AlgoPython.LTM.exp.exp_imputation import Exp_Imputation

# 🔧 Windows checkpoint fix
pathlib.PosixPath = pathlib.PosixPath


# ===== ARGS =====
class Args:
    # basic
    task_name = 'imputation'
    is_training = 0
    is_finetuning = 0
    model_id = 'inference'
    model = 'Timer'
    seed = 0

    # data (unused but required)
    data = 'custom'
    root_path = './dataset/electricity/'
    data_path = 'electricity.csv'
    features = 'M'
    target = 'OT'
    freq = 'h'
    checkpoints = './checkpoints/'
    inverse = False

    # model
    d_model = 256
    n_heads = 8
    e_layers = 4
    d_layers = 1
    d_ff = 512
    factor = 3
    distil = True
    dropout = 0.1
    embed = 'timeF'
    activation = 'gelu'
    output_attention = False

    # optimization (unused)
    num_workers = 0
    itr = 1
    train_epochs = 0
    batch_size = 16
    patience = 3
    learning_rate = 0.001
    des = 'inference'
    loss = 'MSE'
    lradj = 'type1'
    use_amp = False

    # GPU
    use_gpu = True
    gpu = 0
    use_multi_gpu = False
    devices = '0'

    # extra
    stride = 1
    ckpt_path = os.path.abspath('imputegap/wrapper/AlgoPython/LTM/checkpoints/Timer_imputation_1.0.ckpt')
    finetune_epochs = 0
    local_rank = 0

    patch_len = 24
    subset_rand_ratio = 1
    data_type = 'custom'

    decay_fac = 0.75

    cos_warm_up_steps = 100
    cos_max_decay_steps = 60000
    cos_max_decay_epoch = 10
    cos_max = 1e-4
    cos_min = 2e-6

    use_weight_decay = 0
    weight_decay = 0.01

    use_ims = True
    output_len = 96
    output_len_list = None

    train_test = 0

    # sequence
    seq_len = 192
    label_len = 0
    pred_len = 192

    # imputation
    mask_rate = 0.25


# ===== IMPUTER =====
class TimerImputer:
    def __init__(self, args):
        self.args = args

        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        self.device = torch.device(f'cuda:{args.gpu}' if args.use_gpu else 'cpu')

        print("Using device:", self.device)

        # build model via repo
        self.exp = Exp_Imputation(args)
        self.model = self.exp.model.to(self.device)
        self.model.eval()


    def impute(self, data):
        data = data.astype(np.float32)
        # ===== detect missing =====
        missing_global = np.isnan(data)

        # ===== normalize (ignore NaNs) =====
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0) + 1e-6

        norm = (data - mean) / std

        # replace NaNs with 0 for model input
        norm[missing_global] = 0

        seq_len = self.args.seq_len
        pred_len = self.args.pred_len

        result = np.zeros_like(data)
        counts = np.zeros_like(data)

        for i in range(len(data) - seq_len + 1):
            window = norm[i:i + seq_len]
            missing = missing_global[i:i + seq_len]

            # ===== build mask =====
            mask = (~missing).astype(np.float32)

            x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
            mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(self.device)

            B, T, D = x.shape

            # dummy time features
            x_mark_enc = torch.zeros((B, T, 4)).to(self.device)
            x_dec = torch.zeros((B, pred_len, D)).to(self.device)
            x_mark_dec = torch.zeros((B, pred_len, 4)).to(self.device)

            with torch.no_grad():
                outputs = self.model(
                    x,
                    x_mark_enc,
                    x_dec,
                    x_mark_dec,
                    mask=mask_tensor
                )

            outputs = outputs.squeeze(0).cpu().numpy()

            # ===== denormalize =====
            outputs = outputs * std + mean

            # ===== accumulate only missing positions =====
            result[i:i + seq_len][missing] += outputs[missing]
            counts[i:i + seq_len][missing] += 1

        # avoid division by zero
        counts[counts == 0] = 1
        result = result / counts

        # ===== final reconstruction =====
        final = data.copy()
        final[missing_global] = result[missing_global]

        return final, missing_global

