from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class InformerDataset:
    def __init__(
            self,
            forecast_horizon: Optional[int] = 192,
            data_split: str = "train",
            data_stride_len: int = 1,
            task_name: str = "forecasting",
            random_seed: int = 42,
            file_path: str = "data/ETTh1.csv",
            ts_m: Optional[np.ndarray] = None,
            tr_ratio: float = 0.7,
            scaler: Optional[StandardScaler] = None,
            interpolate_missing: bool = True,
            window_starts: Optional[np.ndarray] = None,
            full_series: bool = False,
            return_observed_mask: bool = False,
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
        file_path : str
            Path to the dataset file (.csv or .txt).
        ts_m : np.ndarray, optional
            Directly provide the time series data as a numpy array of shape (T, C).
        tr_ratio : float
            Ratio of the dataset to be used for training.
        """

        self.seq_len = 512
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = file_path
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed
        self.ts_m = ts_m
        self.tr_ratio = tr_ratio
        self.scaler = scaler if scaler is not None else StandardScaler()
        self._external_scaler = scaler is not None
        self.interpolate_missing = interpolate_missing
        self.window_starts = None if window_starts is None else np.asarray(window_starts, dtype=int)
        self.full_series = full_series
        self.return_observed_mask = return_observed_mask

        # Read data
        self._read_data()

    def _get_borders(self):
        n_total = self.length_timeseries_total

        if self.full_series or self.tr_ratio == 0:
            n_train = 0
            n_val = 0
            n_test = n_total
        elif n_total >= 14400 and self.full_file_path_and_name.endswith('ETTh1.csv'):
            n_train = 12 * 30 * 24
            n_val = 4 * 30 * 24
            n_test = 4 * 30 * 24
        else:
            # For smaller datasets, use proportional splits
            n_train = int(n_total * self.tr_ratio)
            n_test = int(n_total * (1 - self.tr_ratio))
            n_val = 0

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        # Ensure we don't go out of bounds or have negative lengths
        test_start = max(0, test_start)
        test_end = min(n_total, test_end)

        train = slice(0, train_end)
        test = slice(test_start, test_end)

        return train, test

    def _read_data(self):
        if self.ts_m is not None:
            df = pd.DataFrame(self.ts_m)
            # Add artificial date column with 1 hour differences
            df.insert(0, 'date', pd.date_range(start='2020-01-01', periods=len(df), freq='h'))
        elif self.full_file_path_and_name.endswith('.txt'):
            import re
            data = []
            with open(self.full_file_path_and_name, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    line_values = []
                    for p in parts:
                        clean_p = re.sub(r'[^0-9.\-eE]', '', p)
                        if clean_p:
                            try:
                                line_values.append(float(clean_p))
                            except ValueError:
                                continue
                    if line_values:
                        data.append(line_values)
            df = pd.DataFrame(data)
            # Add artificial date column with 1 hour differences
            df.insert(0, 'date', pd.date_range(start='2020-01-01', periods=len(df), freq='h'))
        else:
            df = pd.read_csv(self.full_file_path_and_name)

        self.length_timeseries_total = df.shape[0]
        self.n_channels = df.shape[1] - 1

        df.drop(columns=["date"], inplace=True)
        df = df.infer_objects()
        if self.interpolate_missing:
            df = df.interpolate(method="cubic")

        data_splits = self._get_borders()

        train_data = df[data_splits[0]]
        if not self._external_scaler:
            scaler_data = train_data if len(train_data) > 0 else df
            if len(scaler_data) > 0:
                self.scaler.fit(scaler_data.values)

        df_values = df.values
        if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None:
            df_values = self.scaler.transform(df_values)

        if self.data_split == "train":
            self.data = df_values[data_splits[0], :]
        elif self.data_split == "test":
            self.data = df_values[data_splits[1], :]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        if self.window_starts is not None:
            seq_start = int(self.window_starts[index])
        else:
            seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T

            return timeseries, forecast, input_mask

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_start = seq_end - self.seq_len

            timeseries = self.data[seq_start:seq_end, :].T
            if self.return_observed_mask:
                input_mask = (~np.isnan(timeseries)).astype(np.int64)

            return timeseries, input_mask

    def __len__(self):
        if self.task_name == "imputation":
            if self.window_starts is not None:
                return len(self.window_starts)
            if self.length_timeseries < self.seq_len:
                return 0
            return int(np.ceil((self.length_timeseries - self.seq_len) / self.data_stride_len)) + 1
        elif self.task_name == "forecasting":
            return max(0, (
                    self.length_timeseries - self.seq_len - self.forecast_horizon
            ) // self.data_stride_len + 1)
