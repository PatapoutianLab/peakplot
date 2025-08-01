"""
Data input/output operations for peakplot package.

This module handles reading, writing, and converting data files for pressure measurement analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_datafile(filename):
    """
    Read raw data file and return cleaned DataFrame.
    
    Args:
        filename (str): Path to the data file
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Time', 'Pressure', 'Temperature']
    """
    with open(filename, 'r') as f:
        for line in f:
            if line.find('CHN') > 0:
                break
        df = pd.read_csv(f, sep=' ', skipinitialspace=True)
    df.columns = ['Time', 'Pressure', 'Temperature']
    df = df[df['Pressure'] > -50]
    return df


def append_data(df1, df2):
    """
    Append two DataFrames with continuous time indexing.
    
    Args:
        df1 (pd.DataFrame): First DataFrame
        df2 (pd.DataFrame): Second DataFrame to append
        
    Returns:
        pd.DataFrame: Combined DataFrame with continuous time
    """
    df = df2.copy()
    df['Time'] += df1['Time'].iloc[-1]
    return pd.concat([df1, df])


def save_datafile(filename, df, t0_time=0, complib='blosc:lz4'):
    """
    Save 'Time' and 'Pressure' columns of a DataFrame into an HDF5 file.
    Uses table format with compression for partial I/O capability.
    
    Args:
        filename (str): Output HDF5 filename
        df (pd.DataFrame): DataFrame to save
        t0_time (float): Time offset to subtract from timestamps
        complib (str): Compression library to use
    """
    df_save = df.copy()
    df_save["Time"] -= t0_time

    # Use 'table' format so we can do partial reads with "where" queries.
    # Make 'Time' a data_column, so the search is indexed.
    df_save.to_hdf(
        path_or_buf=filename,
        key="df",
        mode="w",
        format="table",
        data_columns=["Time"],   # Critical for partial reads
        complevel=9,
        complib=complib
    )


def npz2hdf(npz_filename, hdf_filename, complib='blosc:lz4'):
    """
    Convert a .npz file (with arrays 'Time' and 'Pressure') to HDF5 format.
    This allows chunked, compressed storage and partial I/O on the 'Time' column.
    
    Args:
        npz_filename (str): Input NPZ filename
        hdf_filename (str): Output HDF5 filename
        complib (str): Compression library to use
    """
    # Load from .npz
    with np.load(npz_filename) as data:
        time_array = data["Time"]
        pressure_array = data["Pressure"]

    # Build a DataFrame
    df = pd.DataFrame({
        "Time": time_array,
        "Pressure": pressure_array
    })

    # Save to HDF
    df.to_hdf(
        path_or_buf=hdf_filename,
        key="df",
        mode="w",
        format="table",
        data_columns=["Time"],
        complevel=9,
        complib=complib
    )


def read_numpydata(filename, time_range=None):
    """
    Read data from an HDF5 file (containing a single table named 'df').
    
    Args:
        filename (str): HDF5 filename to read
        time_range (tuple): Optional (start, end) time range for partial reading
        
    Returns:
        pd.DataFrame: DataFrame with columns ['Time', 'Pressure']
    """
    if time_range is None:
        # Read the entire dataset
        df = pd.read_hdf(filename, key="df")
    else:
        start_time, end_time = time_range
        # Use a 'where' query with the Time column
        query_str = f"Time >= {start_time} & Time <= {end_time}"
        df = pd.read_hdf(filename, key="df", where=query_str)
    return df


def convert_data(save_name, file_list, path_prefix, start_time, birth_time, birth_index):
    """
    Convert and combine multiple data files with visualization.
    
    Args:
        save_name (str): Output filename for combined data
        file_list (list): List of input filenames
        path_prefix (str): Path prefix for input files
        start_time (str): Start time string
        birth_time (str): Birth time string
        birth_index (int): Index of birth file for special plotting
    """
    from .utils import calculate_t0
    from .preprocessing import get_data_range
    from .visualization import plot_range
    
    n_file = len(file_list)
    f, axs = plt.subplots(n_file + 1, 1, figsize=(6, 4 * n_file + 5))
    data_f = None
    t0 = calculate_t0(start_time, birth_time)
    ltime = 0
    
    for idx, file in enumerate(file_list):
        fname = path_prefix + file
        data = read_datafile(fname)
        if idx == birth_index:
            plot_range(data, [t0, t0 + 6 * 3600], axs[-1])
            if data_f is None:
                ltime = 0
            else:
                ltime = data_f['Time'].iloc[-1]

        if idx == 0:
            data_f = data
        else:
            data_f = append_data(data_f, data)
        plot_range(data, [0, 48*3600], axs[idx])
            
    save_datafile(save_name, data_f, ltime + t0)