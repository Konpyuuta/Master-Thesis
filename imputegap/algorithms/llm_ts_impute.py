import numpy as np

from imputegap.wrapper.AlgoPython.LLMTS.llm_integrator_recovery import recov_ts


def llm_ts_impute(incomp_data, params=None):
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

    """

    # Imputation
    recov_data = recov_ts(ts_m=incomp_data)

    return recov_data
