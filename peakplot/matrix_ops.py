"""
Matrix operations and kernel-based analysis for peakplot package.

This module provides functions for multi-animal data analysis and convolution operations.
"""

import numpy as np
import scipy.stats as st


def get_kernel_time(file_list, kernel, threshold, time_range=[0, 3600 * 24], interval='1s'):
    """
    Apply convolution kernel to multiple files and find threshold crossing times.
    
    Args:
        file_list (list): List of data filenames
        kernel (np.ndarray): Convolution kernel
        threshold (float): Threshold for detection
        time_range (tuple): Time range to analyze
        interval (str): Resampling interval
        
    Returns:
        np.ndarray: Array of threshold crossing times for each file
    """
    from .data_io import read_numpydata
    from .preprocessing import resample_data
    
    result = np.zeros(len(file_list))
    
    for idx, fname in enumerate(file_list):
        df = read_numpydata(fname, time_range=time_range)
        time_res = resample_data(df, time_range, interval)
        time_res[time_res < 0] = 0
        val = np.convolve(time_res, kernel, 'valid')
        result[idx] = np.argmax(val > threshold)
    
    return result


def lin_kernel(radius):
    """
    Create a linear (uniform) convolution kernel.
    
    Args:
        radius (int): Kernel radius
        
    Returns:
        np.ndarray: Normalized linear kernel
    """
    return np.ones(2 * radius + 1) / (2 * radius + 1)


def gauss_kernel(radius, scale_factor=0.01):
    """
    Create a Gaussian convolution kernel.
    
    Args:
        radius (int): Kernel radius
        scale_factor (float): Scale factor for Gaussian width
        
    Returns:
        np.ndarray: Normalized Gaussian kernel
    """
    k = st.norm.pdf(np.linspace(-radius, radius, 2 * radius + 1), 
                    scale=radius * scale_factor)
    return k / np.sum(k)