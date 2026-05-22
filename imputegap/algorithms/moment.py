import numpy as np
import torch

from imputegap.wrapper.AlgoPython.MOMENT.MOMENTImputer import to_moment_imputation_format, MOMENTImputer


def moment_impute(incomp_data, params=None):
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

    # core of the algorithm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    '''
    data = np.array([
        [5, 2, np.nan, 4, 1, 7],
        [3, np.nan, 6, 8, 2, 9],
    ], dtype=np.float32)
    print(data.shape)
    data = np.tile(data, (1, 300))  # (2, 1800)'''

    x_enc, input_mask = to_moment_imputation_format(
        incomp_data,
        window_size=incomp_data.shape[1],
        stride=incomp_data.shape[1],
        channels_first=True
    )

    x_enc = x_enc.to(device).float()
    input_mask = input_mask.to(device).long()
    momentimputer = MOMENTImputer()
    momentimputer.init_dataset()
    recov_data = momentimputer.impute(x_enc, input_mask)

    return recov_data
