"""
Peak detection and analysis functions for peakplot package.

This module provides functions for detecting peaks in pressure data and analyzing their properties.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import csv


def get_peak_data(df, half_width=250, prominence=40, distance=1000):
    """
    Extract peak data segments from pressure traces.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Pressure' column
        half_width (int): Half-width of peak segments to extract
        prominence (float): Minimum peak prominence
        distance (int): Minimum distance between peaks
        
    Returns:
        tuple: (peak_data_matrix, peak_indices)
            - peak_data_matrix: Array of shape (n_peaks, 2*half_width)
            - peak_indices: Array of peak positions
    """
    peaks, _ = find_peaks(df['Pressure'], distance=distance, prominence=prominence)
    peak_data = np.zeros([len(peaks), 2 * half_width])
    
    for idx, p in enumerate(peaks):
        peak_data[idx, :] = df['Pressure'][p - half_width:p + half_width]
    
    peak_arg = np.argsort(peak_data[:, half_width])
    peak_plot = peak_data[peak_arg, :]
    return peak_plot, peaks


def get_peaks(df, ini_prominence=8, distance=4000, factor_prominence=15):
    """
    Adaptive peak detection with baseline noise estimation.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Pressure' column
        ini_prominence (float): Initial prominence threshold
        distance (int): Minimum distance between peaks
        factor_prominence (float): Factor to multiply baseline noise for final prominence
        
    Returns:
        np.ndarray: Array of peak indices
    """
    def get_base(p, left_base, right_base, fraction=0.4):
        """Calculate baseline noise level from inter-peak regions."""
        baselines = np.array([])
        rs = np.zeros(len(left_base) - 1)
        
        for idx in range(len(left_base) - 1):
            left = right_base[idx]
            right = left_base[idx + 1]
            length = int((right - left) * fraction)
            baseline = np.copy(p[left + length: right - length])
            
            if len(baseline) < 1:
                continue
                
            baseline -= np.mean(baseline)
            rs[idx] = np.max(baseline) - np.min(baseline)
            
            if rs[idx] < 15:
                baselines = np.concatenate([baselines, baseline])
        
        return np.std(baselines)
    
    peaks, peak_p = find_peaks(df['Pressure'], distance=distance, 
                              prominence=ini_prominence, wlen=distance)
    error = get_base(df['Pressure'], peak_p['left_bases'], peak_p['right_bases'])
    prominence = factor_prominence * error
    peaks = peaks[peak_p['prominences'] >= prominence]
    return peaks


def get_peak_stat(trace, half_width=5, prominence=20, height=40):
    """
    Comprehensive peak statistics analysis.
    
    Args:
        trace (np.ndarray): 1D pressure trace
        half_width (int): Half-width for peak extraction
        prominence (float): Minimum peak prominence
        height (float): Minimum peak height
        
    Returns:
        tuple: (peak_data, peak_indices, widths, aucs)
            - peak_data: Matrix of peak segments
            - peak_indices: Peak positions
            - widths: Peak widths
            - aucs: Peak areas under curve
    """
    from scipy.signal import peak_widths
    
    peaks, properties = find_peaks(trace, distance=2 * half_width, 
                                 prominence=prominence, height=height)
    
    left_bases = properties["left_bases"]
    right_bases = properties["right_bases"]
    widths = right_bases - left_bases + 1
    aucs = np.array([
        np.trapz(trace[left:right + 1]) for left, right in zip(left_bases, right_bases)
    ])
    
    # Filter peaks that are too close to edges or have negative AUC
    filter_mask = np.logical_and(peaks > half_width, peaks < len(trace) - half_width)
    filter_mask = np.logical_and(filter_mask, aucs > 0)
    
    peaks = peaks[filter_mask]
    widths = widths[filter_mask]
    aucs = aucs[filter_mask]
    
    # Extract peak segments
    peak_data = np.zeros([len(peaks), 2 * half_width])
    for idx, p in enumerate(peaks):
        peak_data[idx, :] = trace[p - half_width:p + half_width]
    
    # Sort by peak height
    peak_arg = np.argsort(peak_data[:, half_width])
    peak_plot = peak_data[peak_arg, :]
    
    return peak_plot, peaks, widths, aucs


def save_peak_meta(arr_mat, file_prefix=None, num_peaks=50, prominence=20, height=40):
    """
    Extract and save peak metadata from multiple traces.
    
    Args:
        arr_mat (np.ndarray): Matrix of traces (n_traces, n_timepoints)
        file_prefix (str): Prefix for output CSV files
        num_peaks (int): Maximum number of peaks to keep per trace
        prominence (float): Peak prominence threshold
        height (float): Peak height threshold
        
    Returns:
        tuple: (widths_list, pressures_list, aucs_list)
    """
    widths = []
    pressures = []
    aucs = []
    
    for i in range(arr_mat.shape[0]):
        peak_data, peaks, peak_w, peak_auc = get_peak_stat(
            arr_mat[i, :], prominence=prominence, height=height
        )
        peak_pressures = arr_mat[i, peaks]
        
        # Keep only top N peaks by pressure if specified
        if num_peaks is not None and len(peak_pressures) > num_peaks:
            peak_ranks = np.argsort(peak_pressures)
            peak_pressures = peak_pressures[peak_ranks[-num_peaks:]]
            peak_w = peak_w[peak_ranks[-num_peaks:]]
            peak_auc = peak_auc[peak_ranks[-num_peaks:]]
        
        widths.append(peak_w)
        pressures.append(peak_pressures)
        aucs.append(peak_auc)
    
    # Save to CSV files if prefix provided
    if file_prefix is not None:
        list2csv(widths, file_prefix + '_widths.csv')
        list2csv(pressures, file_prefix + '_heights.csv')  # Note: heights should be pressures
        list2csv(aucs, file_prefix + '_aucs.csv')
    
    return widths, pressures, aucs


def get_stat_data(data, labels, prominence, height, num_peaks=None):
    """
    Extract statistical data from peak analysis for multiple groups.
    
    Args:
        data (list): List of data matrices, one per group
        labels (list): List of group labels
        prominence (float): Peak prominence threshold
        height (float): Peak height threshold
        num_peaks (int): Maximum number of peaks to keep per trace
        
    Returns:
        tuple: (xticks, peak_values, mean_values)
            - xticks: Array of group labels for each data point
            - peak_values: List of peak pressure arrays for each trace
            - mean_values: List of mean peak pressures for each group
    """
    xticks = []
    peak_values = []
    mean_values = []
    
    for idx, trace_matrix in enumerate(data):
        xticks.append(np.repeat(labels[idx], trace_matrix.shape[0]))
        widths, pressures, aucs = save_peak_meta(trace_matrix, num_peaks=num_peaks, 
                                                prominence=prominence, height=height)
        
        # Handle empty peak arrays by setting minimum height
        for i, pressure in enumerate(pressures):
            if len(pressure) == 0:
                pressures[i] = np.array([height])
        
        peak_values += pressures
        mean_values.append(np.array([np.mean(p) for p in pressures]))
    
    xticks = np.concatenate(xticks)
    return xticks, peak_values, mean_values


def list2csv(data_list, filename):
    """
    Save a list of arrays to CSV file.
    
    Args:
        data_list (list): List of arrays to save
        filename (str): Output CSV filename
    """
    with open(filename, 'w', newline="") as f:
        writer = csv.writer(f)
        for row in data_list:
            writer.writerow(row)