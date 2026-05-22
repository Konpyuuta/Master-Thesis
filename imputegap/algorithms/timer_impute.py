import os
import pathlib
import random

import numpy as np
import torch

from wrapper.AlgoPython.LTM.timer_imputer import TimerImputer

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

def timer_impute(incomp_data, params=None):
    """
    Impute NaN values with the mean value of the time series.

    Parameters
    ----------
    incomp_data : numpy.ndarray
        The input time series with contamination (missing values represented as NaNs).
    params : dict, optional
        Optional parameters for the algorithm. If None, the minimum value from the contamination is used (default is None).

    Returns
    -------
    numpy.ndarray
        The imputed matrix where NaN values have been replaced with the mean value from the time series.

    Notes
    -----
    This function finds the non-NaN value in the time series and replaces all NaN values with this mean value.
    It is a simple imputation technique for filling missing data points in a dataset.

    Example
    -------
        >>> incomp_data = np.array([[5, 2, np.nan], [3, np.nan, 6]])
        >>> recov_data = mean_impute(incomp_data)
        >>> print(recov_data)
        array([[5., 2., 4.],
               [3., 4., 6.]])

    """

    args = Args()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    imputer = TimerImputer(args)

    # ===== YOUR DATA =====
    # data = np.random.rand(300, 5)

    recov_data, mask = imputer.impute(incomp_data)

    print(recov_data[0, 1:80])

    return recov_data
