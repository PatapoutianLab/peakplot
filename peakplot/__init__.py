"""
Peakplot: A package for pressure measurement analysis and visualization.

This package provides tools for analyzing pressure data, detecting peaks,
and visualizing birth timing patterns in biological research.

Modules:
- data_io: Data input/output operations
- preprocessing: Data filtering and resampling
- peak_analysis: Peak detection and statistical analysis
- visualization: Plotting functions for data and results
- matrix_ops: Multi-file analysis and convolution operations
- utils: Utility functions for time calculations and data export
"""

# Import commonly used functions for easy access
from .data_io import (
    read_datafile, 
    save_datafile, 
    read_numpydata,
    append_data,
    npz2hdf,
    convert_data
)

from .preprocessing import (
    get_data_range,
    resample_data,
    get_plot_mat
)

from .peak_analysis import (
    get_peaks,
    get_peak_data,
    get_peak_stat,
    save_peak_meta,
    get_stat_data,
    list2csv
)

from .visualization import (
    plot_range,
    plot_peaks,
    plot_birth,
    plot_birthtimes,
    plot_all_press,
    plot_bar_with_pies,
    make_trace_plot
)

from .matrix_ops import (
    get_kernel_time,
    lin_kernel,
    gauss_kernel
)

from .utils import (
    calculate_t0,
    time_to_seconds,
    time_interval_to_seconds,
    get_mean_birth_interval,
    birth2zeit
)

__version__ = "1.0.0"
__author__ = "GitHub@Xbar @RenhaoL"
