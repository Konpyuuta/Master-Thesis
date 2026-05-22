'''import subprocess
import random


class UniTSImputer():
    def __init__(
        self,
        model_name="UniTS_zeroshot",
        exp_name="UniTS_zeroshot_pretrain_x64",
        wandb_mode="offline",
        ptune_name="zeroshot_newdata",
        d_model=128,
        master_port=None,
    ):
        self.model_name = model_name
        self.exp_name = exp_name
        self.wandb_mode = wandb_mode
        self.ptune_name = ptune_name
        self.d_model = d_model

        self.master_port = master_port or random.randint(1000, 9999)

    def build_command(self):
        cmd = [
            "torchrun",
            "--nnodes", "1",
            "--master_port", str(self.master_port),
            "run.py",
            "--is_training", "0",
            "--model_id", self.exp_name,
            "--model", self.model_name,
            "--prompt_num", "10",
            "--patch_len", "16",
            "--stride", "16",
            "--batch_size", "1",
            "--task_name", "imputation",
            "--subsample_pct", "0.001",
            "--e_layers", "3",
            "--d_model", str(self.d_model),
            "--des", "Exp",
            "--debug", self.wandb_mode,
            "--project_name", self.ptune_name,
            "--pretrained_weight", "units_x128_pretrain_checkpoint.pth",
            "--task_data_config_path", "data_provider/imputation.yaml",
        ]
        return cmd

    def run(self):
        cmd = self.build_command()

        print("Running command:")
        print(" ".join(cmd))

        subprocess.run(cmd, check=True)'''

import torch
from types import SimpleNamespace

from wrapper.AlgoPython.UniTS.models.UniTS_zeroshot import Model

# Define arguments ..
args = SimpleNamespace(
    d_model=128,
    is_training=0,
    n_heads=4,
    e_layers=3,
    patch_len=16,
    stride=16,
    dropout=0.1,
    prompt_num=10
)

# Configuration ...
configs_list = [
    ("my_task", {
        "task_name": "imputation",
        "enc_in": 12,
        "seq_len": 256,
        "features": "M",
        "loss": "MSE"
    })
]

def load_units_model(args, configs_list, checkpoint_path, device="cuda"):
    model = Model(args, configs_list).to(device)

    print(f"Loading checkpoint: {checkpoint_path}")

    ckpt_raw = torch.load(checkpoint_path, map_location=device)

    # handle pretrain checkpoints
    if isinstance(ckpt_raw, dict) and "student" in ckpt_raw:
        state_dict = ckpt_raw["student"]
    else:
        state_dict = ckpt_raw

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v

    msg = model.load_state_dict(new_state_dict, strict=False)
    print("Load result:", msg)

    model.eval()
    return model




# ---- wrapper ----
class UniTSImputer:
    def __init__(self, task_id=0, device="cuda"):
        # ---- model ----
        model = load_units_model(args, configs_list, "units_x128_pretrain_checkpoint.pth")
        model.eval()
        self.model = model.to(device)
        self.task_id = task_id
        self.device = device

    def impute(self, x_raw, mas):
        # x_raw expected: (C, T) OR (T, C)

        if x_raw.ndim == 2:
            x = x_raw.T.unsqueeze(0)  # (1, T, C)
        elif x_raw.ndim == 3:
            x = x_raw  # already batched
        else:
            raise ValueError(f"Unexpected shape: {x_raw.shape}")

        x = x.to(self.device)

        mask = ~torch.isnan(x)
        mask = mask.float()

        x = torch.nan_to_num(x, nan=0.0)

        x_mark = torch.zeros(x.shape[0], x.shape[1], 1).to(self.device)

        with torch.no_grad():
            out = self.model(
                x_enc=x,
                x_mark_enc=x_mark,
                mask=mask,
                task_id=self.task_id,
                task_name="imputation"
            )

        return out.squeeze(0).T.cpu().numpy()
