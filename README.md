# Peakplot

A Python package for analyzing telemetry data from PhysioTel HD implants manufactured by Data Science International (DSI). The package is specialized for intrauterine pressure measurements but can be adapted for other physiological signal analysis applications.

## Overview

Peakplot provides comprehensive tools for processing, analyzing, and visualizing pressure measurement data. It includes functions for peak detection, statistical analysis, birth timing patterns, and multi-animal comparative studies. The package is designed to handle large datasets and supports various data formats commonly used in physiological research.

## Features

- **Data I/O**: Read and write data in multiple formats (HDF5, CSV, NPZ)
- **Peak Detection**: Adaptive peak detection with noise estimation
- **Statistical Analysis**: Comprehensive peak statistics and birth timing analysis
- **Visualization**: Rich plotting capabilities for data exploration and publication-ready figures
- **Multi-animal Analysis**: Batch processing and comparative analysis across multiple subjects
- **Time Series Processing**: Data filtering, resampling, and temporal analysis

## Modules

### `data_io`
Handles data input/output operations including:
- Reading raw data files from DSI telemetry systems
- Converting between data formats (NPZ to HDF5, CSV export)
- Appending multiple data files with continuous time indexing
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
- Kernel-based analysis for birth timing detection
- Linear and Gaussian kernel generation
- Batch processing across multiple animals
- Threshold-based event detection

### `utils`
Utility functions for common operations:
- Time format conversions and calculations
- Birth interval statistical analysis
- Zeitgeber time conversions for circadian analysis
- Data export helpers

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd peakplot

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

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

## Requirements

- Python 3.6+

All package dependencies are listed in `requirements.txt` and can be installed using:
```bash
pip install -r requirements.txt
```

Core dependencies include:
- NumPy (≥1.19.0)
- Pandas (≥1.3.0)
- SciPy (≥1.7.0)
- Matplotlib (≥3.3.0)
- PyTables (≥3.6.0) - for HDF5 support

## Data from publication

For data package from the publication, please download from Zenodo doi.10.5281/zenodo.16699930

## License

The GNU General Public License v3.0


