# Copyright (c) 2024-present, Royal Bank of Canada.
# Copyright (c) 2021 THUML @ Tsinghua University
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the TimesNet (https://arxiv.org/abs/2210.02186) implementation
# from https://github.com/thuml/Time-Series-Library by THUML @ Tsinghua University
####################################################################################

import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from imputegap.wrapper.AlgoPython.LLMTS.utils.timefeatures import time_features
from imputegap.wrapper.AlgoPython.LLMTS.data_provider.m4 import M4Dataset, M4Meta
from imputegap.wrapper.AlgoPython.LLMTS.data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings("ignore")


class ImputeGAP_Dataset(Dataset):

    # Add ndarray with time series as parameter ..
    # Preprocess
    def __init__(
        self,
        time_series_data=None,
        flag="train",
        size=None,
        features="S",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.time_series_data = time_series_data

        self.__read_data__()

    def __read_data__(self):
        """
        data_array: np.ndarray containing sensor data (without date column)
        Assumes target is the last column of the array.
        """
        self.scaler = StandardScaler()
        data_array = self.time_series_data

        # 1. Generate the date column manually (1-hour intervals)
        dates = pd.date_range(
            start='2016-07-01 02:00:00',
            periods=len(data_array),
            freq='1H'
        )

        # 2. Convert ndarray to DataFrame
        df_raw = pd.DataFrame(data_array)

        # 3. Handle column names to support the original logic
        # We name the last column as 'target' and others as features
        feature_count = data_array.shape[1] - 1
        cols_names = [f'feat_{i}' for i in range(feature_count)]
        self.target = 'target'
        df_raw.columns = cols_names + [self.target]

        # 4. Insert the generated date at the beginning
        df_raw.insert(0, 'date', dates)

        # --- Original Logic Starts Here ---
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]: border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        # Dates are already datetime objects from pd.date_range, but this is safe:
        df_stamp["date"] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            # time_features is an external utility usually found in Informer/Autoformer
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)











