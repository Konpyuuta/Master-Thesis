import numpy as np
import torch

from wrapper.AlgoPython.UniTS.UniTSImputer import UniTSImputer


def preprocess_data(data):
    x = data
    x = torch.tensor(x, dtype=torch.float32)
    x = x.T.unsqueeze(0)
    mask = ~torch.isnan(x)
    mask = mask.float()

    return x, mask


def uni_ts(incomp_data, params=None):
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data, mask = preprocess_data(incomp_data)

    '''
    data = np.array([
        [5, 2, np.nan, 4, 1, 7],
        [3, np.nan, 6, 8, 2, 9],
    ], dtype=np.float32)
    print(data.shape)
    data = np.tile(data, (1, 300))  # (2, 1800)'''


    print("UniTS Data shape:")
    print(incomp_data.shape)
    print(incomp_data)

    # ---- create imputer ----
    imputer = UniTSImputer()

    # ---- run ----
    recov_data = imputer.impute(data, mask)

    print(recov_data[0, 1:80])

    return recov_data
