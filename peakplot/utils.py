"""
Utility functions for peakplot package.

This module provides helper functions for time calculations, data export, and statistical analysis.
"""

import numpy as np
from datetime import datetime, timedelta


def calculate_t0(time_start, time_end, date_format='%m/%d %H:%M'):
    """
    Calculate time difference between two time strings.
    
    Args:
        time_start (str): Start time string
        time_end (str): End time string
        date_format (str): Format string for parsing times
        
    Returns:
        float: Time difference in seconds
    """
    t1 = datetime.strptime(time_start, date_format)
    t2 = datetime.strptime(time_end, date_format)
    return (t2 - t1).total_seconds()


def time_to_seconds(time_str):
    """
    Convert a time string in %H:%M format to seconds since midnight.
    
    Args:
        time_str (str): Time string in "HH:MM" format
        
    Returns:
        int: Total seconds since midnight
    """
    time_obj = datetime.strptime(time_str, "%H:%M")
    total_seconds = time_obj.hour * 3600 + time_obj.minute * 60
    return total_seconds


def time_interval_to_seconds(time_interval):
    """
    Convert time interval string to seconds.
    
    Args:
        time_interval (str): Time interval (e.g., "1s", "5min", "2h", "1d")
        
    Returns:
        float: Interval in seconds
    """
    # Extract numeric value and unit part from the input string
    numeric_part = int(''.join(filter(str.isdigit, time_interval)))
    unit_part = ''.join(filter(str.isalpha, time_interval))

    # Define the conversion logic using timedelta
    if unit_part == 's':
        return numeric_part
    elif unit_part == 'min':
        return timedelta(minutes=numeric_part).total_seconds()
    elif unit_part == 'h':
        return timedelta(hours=numeric_part).total_seconds()
    elif unit_part == 'd':
        return timedelta(days=numeric_part).total_seconds()
    else:
        raise ValueError("Invalid time interval format. Please use 's', 'min', 'h', or 'd'.")


def get_mean_birth_interval(birth_time_list, end_time='1/20 12:00', completed=None):
    """
    Calculate mean birth intervals for each animal.
    
    Args:
        birth_time_list (list): List of birth time lists for each animal
        end_time (str): End time for incomplete births
        completed (list): Boolean list indicating completed births
        
    Returns:
        np.ndarray: Mean intervals in seconds for each animal
    """
    result = np.zeros(len(birth_time_list))
    
    for idx, animal in enumerate(birth_time_list):
        if completed is not None and (not completed[idx]):
            result[idx] = calculate_t0(animal[0], end_time) / len(animal)
        else:
            result[idx] = calculate_t0(animal[0], animal[-1]) / (len(animal) - 1)
    
    return result


def birth2zeit(birth_time_list, dawn_time='1/19 6:00'):
    """
    Convert birth times to zeitgeber time (hours relative to dawn).
    
    Args:
        birth_time_list (list): List of birth time lists
        dawn_time (str): Dawn time reference
        
    Returns:
        np.ndarray: Zeitgeber times in hours
    """
    zeit_list = []
    for birth_time in birth_time_list:
        dt = calculate_t0(dawn_time, birth_time[0]) / 3600
        zeit_list.append(dt)
    return np.array(zeit_list)