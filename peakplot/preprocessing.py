"""
Data preprocessing and filtering operations for peakplot package.

This module handles data filtering, resampling, and transformation operations.
"""

import pandas as pd
import numpy as np


def get_data_range(df, time_range):
    """
    Filter DataFrame to a specific time range.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Time' column
        time_range (tuple): (start_time, end_time) tuple
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    df2 = df[df['Time'] > time_range[0]]
    df2 = df2[df2['Time'] < time_range[1]]
    return df2


def resample_data(df, time_range=None, interval='1s', method='max'):
    """
    Resample data to a different time interval using specified aggregation method.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'Time' and 'Pressure' columns
        time_range (tuple): Optional time range to filter data first
        interval (str): Resampling interval (e.g., '1s', '1min')
        method (str): Aggregation method ('max' or 'mean')
        
    Returns:
        np.ndarray: Resampled pressure values
    """
    assert method in ['max', 'mean'], f"Unsupported method {method}"
    
    if time_range is None:
        df_in = df.copy()
    else:
        df_in = get_data_range(df, time_range)
    
    df_in['tidx'] = pd.to_datetime(df_in['Time'], unit='s')
    df_in.set_index('tidx', inplace=True)

    if method == 'max':
        resample_df = df_in.resample(interval).max()
    elif method == 'mean':
        resample_df = df_in.resample(interval).mean()
    
    resample_df = resample_df.fillna(0)    
    return resample_df['Pressure'].to_numpy()


def get_plot_mat(file_list, birth_time=None, time_range=None, day_range=None, 
                resample_method='max', interval='1s'):
    """
    Generate a matrix of resampled data from multiple files for visualization.
    
    Args:
        file_list (list): List of HDF5 filenames
        birth_time (list): List of birth times for each file (optional)
        time_range (tuple): Fixed time range (start, end) for all files
        day_range (tuple): Day range for birth-relative timing
        resample_method (str): Resampling method ('max' or 'mean')
        interval (str): Resampling interval
        
    Returns:
        np.ndarray: Matrix of shape (n_files, n_timepoints)
    """
    from .data_io import read_numpydata
    from .utils import calculate_t0
    
    assert (time_range is not None) or (day_range is not None), "Need to provide a valid time range."
    
    calc_range = False
    interval_sec = pd.to_timedelta(interval).total_seconds()
    
    if time_range is None:
        assert birth_time is not None, "Need to have birth time to calculate days"
        time_len = int(calculate_t0(day_range[0], day_range[1]) / interval_sec) + 1
        assert len(file_list) == len(birth_time), "Each data file needs a birth time."
        calc_range = True
    else:
        time_len = int((time_range[1] - time_range[0]) / interval_sec) + 1
    
    result = np.zeros([len(file_list), time_len])
    
    for idx, fname in enumerate(file_list):
        if calc_range:
            t0 = calculate_t0(day_range[0], birth_time[idx][0])
            t1 = calculate_t0(day_range[1], birth_time[idx][0])
            assert t0 > t1, "Day range should be positive"
            time_range = [-t0, -t1]
        
        df = read_numpydata(fname, time_range=time_range)
        row_data = resample_data(df, interval=interval, method=resample_method)
        result[idx, -len(row_data):] = row_data
    
    result[np.isnan(result)] = 0
    return result