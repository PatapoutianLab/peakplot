# Peakplot

A Python package for analyzing telemetry data from PhysioTel HD implants manufactured by Data Science International (DSI). The package is specialized for intrauterine pressure measurements but can be adapted for other physiological signal analysis applications.

## Overview

Peakplot provides basic tools for processing, analyzing, and visualizing pressure measurement data. It includes functions for peak detection, statistical analysis, and visualization. The package is designed to handle large datasets and can significantly reduce space requirement from DSI Ponemah exported files.

## Features

- **Data I/O**: Read and write data from DSI exports
- **Peak Detection**: Adaptive peak detection with noise estimation
- **Statistical Analysis**: Comprehensive peak statistics and birth timing analysis
- **Visualization**: Rich plotting capabilities for data exploration and publication-ready figures
- **Time Series Processing**: Data filtering, resampling, and temporal analysis

## Modules

### `data_io`
Handles data input/output operations including:
- Importing raw data files from DSI telemetry systems
- Combining multiple data files with continuous time indexing
- Saving processed data with compression

### `preprocessing`
Data filtering and preparation functions:
- Time range filtering
- Data resampling with configurable intervals and aggregation methods
- Matrix preparation for multi-file analysis
- Data quality filtering and cleaning

### `peak_analysis`
Core peak detection and statistical analysis:
- Adaptive peak detection with baseline noise estimation
- Peak data extraction and segmentation
- Statistical analysis of peak properties (amplitude, frequency, timing)
- Birth timing analysis and interval calculations
- Data export functions for statistical software

### `visualization`
Comprehensive plotting functions:
- Time series plotting with customizable ranges
- Peak overlay visualization
- Birth timing and pattern analysis plots
- Multi-panel figures with statistical summaries
- Publication-ready plots with customizable styling

### `matrix_ops`
Multi-file analysis and convolution operations:
- Kernel-based analysis for threshold-based event detection
- Linear and Gaussian kernel generation

### `utils`
Utility functions for common operations:
- Time format conversions and calculations
- Birth interval statistical analysis
- Zeitgeber time conversions for circadian analysis
- Data export helpers

## Running the code

```bash
# Clone the repository
git clone https://github.com/Xbar/peakplot
cd peakplot

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Dependencies
To run the code you will need Python>=3.6, and the following packages installed
- NumPy (≥1.19.0)
- Pandas (≥1.3.0)
- SciPy (≥1.7.0)
- Matplotlib (≥3.3.0)
- PyTables (≥3.6.0) - for HDF5 support

## Usage

```python
import peakplot as pp

# Read data file
df = pp.read_datafile('data.txt')

# Detect peaks
peaks = pp.get_peaks(df)

# Analyze peak statistics
peak_stats = pp.get_peak_stat(peaks)

# Create visualization
pp.plot_peaks(df, peaks)
```

## Data Format

The package expects ASCII data files exported from DSI Ponemah with columns:
- `Time`: Time in seconds
- `Pressure`: Pressure measurements in mmHg
- `Temperature`: Temperature measurements (optional)

## Data from publication

For data package from the publication, please download from Zenodo doi.10.5281/zenodo.16699930

## License

The GNU General Public License v3.0


