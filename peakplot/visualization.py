"""
Visualization functions for peakplot package.

This module provides plotting functions for pressure data, peaks, and birth timing analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_range(df, time_range, ax=None):
    """
    Plot pressure data over a specified time range.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Time' and 'Pressure' columns
        time_range (tuple): (start_time, end_time) tuple
        ax (matplotlib.axes.Axes): Optional axes object to plot on
    """
    from .preprocessing import get_data_range
    
    df2 = get_data_range(df, time_range)
    if ax is None:
        plt.plot(df2['Time'], df2['Pressure'], 'k-')
    else:
        ax.plot(df2['Time'], df2['Pressure'], 'k-')


def plot_peaks(df, peaks, time_range=None):
    """
    Plot pressure data with peak markers.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Time' and 'Pressure' columns
        peaks (np.ndarray): Array of peak indices
        time_range (tuple): Optional time range for plotting
    """
    from .preprocessing import get_data_range
    
    plot_df = df
    peak_df = pd.DataFrame({
        'Time': np.array(df['Time'])[peaks],
        'Pressure': np.array(df['Pressure'])[peaks]
    })
    
    if time_range is not None:
        plot_df = get_data_range(df, time_range)
        peak_df = get_data_range(peak_df, time_range)
    
    plt.plot(plot_df['Time'], plot_df['Pressure'], 'k-')
    plt.plot(peak_df['Time'], peak_df['Pressure'], 'x')


def plot_birth(dsi_data, birth_times, time_range=[-48 * 3600, 48 * 3600], 
               y_range=[-50, 450], highlight_range=[-3600, 7200],
               highlight_color='#f5ae78', birth_color='#00aeef', 
               day_cycle=[6, 18], ylabel='Pressure (mmHg)'):
    """
    Plot birth timing data with day/night cycle visualization.
    
    Args:
        dsi_data (pd.DataFrame): Pressure data
        birth_times (list): List of birth time strings
        time_range (tuple): Time range for plotting
        y_range (tuple): Y-axis range
        highlight_range (tuple): Time range to highlight
        highlight_color (str): Color for highlighted region
        birth_color (str): Color for birth markers
        day_cycle (tuple): (dawn_hour, dusk_hour) for day/night cycle
        ylabel (str): Y-axis label
    """
    from .utils import calculate_t0, time_to_seconds
    from .preprocessing import get_data_range
    
    def resample_data_min(df, time_range, interval='1min'):
        """Resample data for efficient plotting."""
        df_in = get_data_range(df, time_range)
        df_in['tidx'] = pd.to_datetime(df_in['Time'], unit='s')
        df_in.set_index('tidx', inplace=True)
        resample_df = df_in.resample(interval).max()
        resample_df.fillna(0)
        return resample_df[['Time', 'Pressure']]
    
    btimes = np.zeros(len(birth_times))
    for idx, t in enumerate(birth_times):
        btimes[idx] = calculate_t0(birth_times[0], t)
        
    time0 = birth_times[0].split(' ')[1]
    time0 = time_to_seconds(time0)
    day_light = np.concatenate([
        np.zeros(day_cycle[0] * 3600, dtype=int), 
        np.ones((day_cycle[1] - day_cycle[0]) * 3600, dtype=int), 
        np.zeros((24 - day_cycle[1]) * 3600, dtype=int)
    ])

    num_days = int(np.floor((time_range[1] - time_range[0]) / (24 * 3600)) + 3)
    light_bar = np.tile(day_light, num_days)

    df = resample_data_min(dsi_data, time_range, interval='10s')
    cmap = ListedColormap(['black', 'white'])
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, 
                                gridspec_kw={'height_ratios': [0.05, 0.95],
                                           'hspace': 0.01})
    
    ax2.plot(df['Time'], df['Pressure'], 'k-')
    ax2.set_ylim(y_range)
    ax2.axvspan(highlight_range[0], highlight_range[1], color=highlight_color, alpha=0.5)
    ax2.vlines(btimes, y_range[0], y_range[1], colors=birth_color)
    ax2.set_ylabel(ylabel)
    ax2.spines['top'].set_visible(False)

    xlim = ax2.get_xlim()
    
    t_start = int(np.round((xlim[0]))) % (24 * 3600) + time0
    t_end = int(xlim[1] - xlim[0]) + t_start
    light_bar = light_bar[t_start:t_end]
    light_bar = np.reshape(light_bar, [1, len(light_bar)])

    ax1.imshow(light_bar, aspect='auto', cmap=cmap, 
               extent=[xlim[0], xlim[1], 0, 1])
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.spines['right'].set_visible(True)
    ax1.spines['left'].set_visible(True)


def plot_bw_bar(ax, time_range, day_cycle=[6, 18]):
    """
    Plot black/white bar showing day/night cycle.
    
    Args:
        ax (matplotlib.axes.Axes): Axes to plot on
        time_range (tuple): Time range strings
        day_cycle (tuple): (dawn_hour, dusk_hour)
    """
    from .utils import calculate_t0, time_to_seconds
    
    cmap = ListedColormap(['black', 'white'])
    day_light = np.concatenate([
        np.zeros(day_cycle[0] * 3600, dtype=int), 
        np.ones((day_cycle[1] - day_cycle[0]) * 3600, dtype=int), 
        np.zeros((24 - day_cycle[1]) * 3600, dtype=int)
    ])

    num_sec = int(calculate_t0(time_range[0], time_range[1]))
    num_days = int(np.floor(num_sec / (24 * 3600)) + 2)
    light_bar = np.tile(day_light, num_days)
    t_start = int(time_to_seconds(time_range[0].split(' ')[1]))
    light_bar = light_bar[t_start:t_start + num_sec]
    light_bar = light_bar.reshape([1, len(light_bar)])
    
    ax.imshow(light_bar, aspect='auto', cmap=cmap, interpolation='nearest',
              extent=[0, num_sec, 0, 1])
    ax.set_yticks([])
    ax.set_xticks([])


def plot_birthtimes(birth_times, img_name, time_range=['1/18 18:00', '1/20 12:00'],
                   birth_color='#00aeef', size_wh=(14, 7)):
    """
    Plot birth times for multiple animals with day/night cycle.
    
    Args:
        birth_times (list): List of birth time lists for each animal
        img_name (str): Output image filename
        time_range (tuple): Time range for plotting
        birth_color (str): Color for birth markers
        size_wh (tuple): Figure size (width, height)
    """
    from .utils import calculate_t0
    
    num_bars = len(birth_times)
    height_ratios = np.ones(num_bars + 1)
    height_ratios[0] = 0.5
    height_ratios /= np.sum(height_ratios)
    margins = 0
    total_height = size_wh[1] * sum(height_ratios) + 2 * margins
    total_width = size_wh[0] + 2 * margins
    
    f, axs = plt.subplots(num_bars + 1, 1,
                         gridspec_kw={'height_ratios': height_ratios,
                                     'hspace': 0.01},
                         figsize=(total_width, total_height))
    
    max_x = calculate_t0(time_range[0], time_range[1])
    for idx in range(num_bars):
        for t in birth_times[idx]:
            pos = calculate_t0(time_range[0], t)
            axs[idx+1].vlines(pos, 0, 1, colors=birth_color, linewidth=4)
        axs[idx+1].set_xlim([0, max_x])
        axs[idx+1].set_ylim([0, 1])
        axs[idx+1].set_xticks([])
        axs[idx+1].set_yticks([])        
    
    plot_bw_bar(axs[0], time_range)
    f.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig(img_name, dpi=300)


def plot_all_press(file_list, img_name, time_range=[-3600, 20 * 3600], 
                   tick_interval=5 * 3600, interval='1min', cmap='Purples'):
    """
    Plot pressure heatmaps for multiple files.
    
    Args:
        file_list (list): List of data filenames
        img_name (str): Output image filename
        time_range (tuple): Time range for plotting
        tick_interval (int): Interval for x-axis ticks
        interval (str): Resampling interval
        cmap (str): Colormap name
    """
    from .data_io import read_numpydata
    from .preprocessing import resample_data
    from .utils import time_interval_to_seconds
    
    f, axs = plt.subplots(len(file_list), 1)

    for idx, fname in enumerate(file_list):
        df = read_numpydata(fname, time_range=time_range)
        values = resample_data(df, interval=interval)
        axs[idx].imshow(values.reshape([1, len(values)]),
                       cmap=cmap, vmin=0, vmax=150, 
                       aspect='auto', interpolation='nearest')
        axs[idx].set_yticks([])
        axs[idx].set_xticks([])
        
        if idx == len(file_list) - 1:
            last_tick = np.floor(time_range[1] / tick_interval) + 1
            ticks = np.arange(int(last_tick))
            ticks = ticks * tick_interval / time_interval_to_seconds(interval)
            ticks -= time_range[0] / time_interval_to_seconds(interval)
            axs[idx].set_xticks(ticks)
    
    plt.savefig(img_name, dpi=300)


def plot_bar_with_pies(data, output_file=None, treatment_labels=None, outcome_labels=None):
    """
    Plot grouped bar chart with pie chart overlays for treatment comparison.
    
    Args:
        data (np.ndarray): 2D array of shape (n_treatments, n_outcomes)
        output_file (str): Optional output filename
        treatment_labels (list): Labels for treatments on x-axis
        outcome_labels (list): Labels for outcomes in legend
    """
    n_treatments, n_outcomes = data.shape
    
    # Provide defaults if labels not supplied
    if treatment_labels is None:
        treatment_labels = [f"Treatment {i+1}" for i in range(n_treatments)]
    if outcome_labels is None:
        outcome_labels = [f"Outcome {j+1}" for j in range(n_outcomes)]
    
    # Create x-position array for the n_treatments
    x_positions = np.arange(n_treatments)
    
    # Width of each bar (make sure the cluster isn't too wide)
    bar_width = 0.7 / n_outcomes
    
    # Pick a color scheme for the outcomes
    colors = plt.cm.gray(np.linspace(0.1, 0.9, n_outcomes))
    
    # Create a figure/axes
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot grouped bars
    # Each outcome is offset horizontally within the cluster
    for j in range(n_outcomes):
        # Calculate offset so that bars in the same cluster remain side-by-side
        x_offset = x_positions + (j - (n_outcomes - 1)/2.0) * bar_width
        
        ax.bar(
            x_offset,
            data[:, j],
            width=bar_width,
            color=colors[j],
            label=outcome_labels[j],
            zorder=2,  # Ensures bars plot above grid lines
        )
    
    # Now add a small pie chart above each cluster
    max_y = np.max(data)
    for i in range(n_treatments):
        # Compute the center x-position of the cluster
        cluster_center = x_positions[i]
        
        # Figure out a good vertical position for the pie
        # For example, place it at a small margin above the maximum bar for that cluster
        pie_center_y = max_y + 3  # You can adjust the +3 as you like
        
        # Create a small inset axis at the (cluster_center, pie_center_y) coordinate
        axins = inset_axes(
            ax,
            width=0.7,   # size: 70% of the parent Axes
            height=0.7,  # size: 70% of the parent Axes
            loc='center',
            bbox_to_anchor=(cluster_center, pie_center_y),
            bbox_transform=ax.transData,  # interpret bbox in data coordinates
            borderpad=0,
        )
        
        # Draw the pie; no labels in the slices themselves
        axins.pie(
            data[i, :],
            colors=colors,
            startangle=90,
            labels=None
        )
        
        # Keep the pie chart circle-shaped
        axins.set_aspect('equal')
    
    # Beautify the main Axes
    ax.set_xticks(x_positions)
    ax.set_xticklabels(treatment_labels)
    ax.set_ylabel("Count")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(False)
    
    # Build legend: put it somewhere that doesn't overlap
    ax.legend(title="Outcomes", loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    if output_file is not None:
        plt.savefig(output_file, dpi=300)


def make_trace_plot(datafile, birth_time, output_name, 
                   day_range=['1/17 12:00', '1/20 12:00'],
                   highlight_range=[0, 7200], birth_color='#37b34a'):
    """
    Create individual trace plot with birth timing.
    
    Args:
        datafile (str): Data filename
        birth_time (list): List of birth times
        output_name (str): Output filename
        day_range (tuple): Day range for plotting
        highlight_range (tuple): Time range to highlight
        birth_color (str): Color for birth markers
    """
    from .data_io import read_numpydata
    from .utils import calculate_t0
    
    t0 = calculate_t0(day_range[0], birth_time[0])
    t1 = calculate_t0(birth_time[0], day_range[1])
    df = read_numpydata(datafile, time_range=[-t0, t1])
    plot_birth(df, birth_time, [-t0, t1], 
              birth_color=birth_color, highlight_range=highlight_range)
    plt.savefig(output_name, dpi=300)