

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class CustomDataset:

    _freq = None

    def __init__(
        self,
        forecast_horizon: Optional[int] = 192,
        freq="h",
        data_split: str = "train",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        """

        self.seq_len = 512
        self._freq = freq
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = "imputegap/wrapper/AlgoPython/MOMENT/ETTh1_HULL.csv"
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed

        # Read data
        self._read_data()

    def _get_borders(self):
        n_train = 12 * 30 * 24
        n_val = 4 * 30 * 24
        n_test = 4 * 30 * 24

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        test = slice(test_start, test_end)

        return train, test

    def _read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1

        #df.drop(columns=["date"], inplace=True)
        df = df.infer_objects(copy=False).interpolate(method="cubic")

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]
        self.scaler.fit(train_data.values)
        df = self.scaler.transform(df.values)

        if self.data_split == "train":
            self.data = df[data_splits[0], :]
        elif self.data_split == "test":
            self.data = df[data_splits[1], :]

        self.length_timeseries = self.data.shape[0]

        # If the timeseries is shorter than the sequence length, pad it with zeros
        if self.length_timeseries < self.seq_len:
            pad_len = self.seq_len - self.length_timeseries
            self.data = np.pad(self.data, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            self.length_timeseries = self.data.shape[0]
            # We should also keep track of the valid mask
            self.valid_mask = np.zeros(self.seq_len)
            self.valid_mask[:self.length_timeseries - pad_len] = 1
        else:
            self.valid_mask = np.ones(self.length_timeseries)

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        
        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T
            input_mask = np.ones(self.seq_len)

            return timeseries, forecast, input_mask

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_start = max(0, seq_end - self.seq_len)

            timeseries = self.data[seq_start:seq_end, :].T
            
            if hasattr(self, 'valid_mask') and len(self.valid_mask) == self.length_timeseries:
                input_mask = self.valid_mask[seq_start:seq_end]
            else:
                input_mask = np.ones(self.seq_len)

            return timeseries, input_mask

    def __len__(self):
        if self.task_name == "imputation":
            return max(0, (self.length_timeseries - self.seq_len) // self.data_stride_len + 1)
        elif self.task_name == "forecasting":
            return max(0, (
                self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1)
