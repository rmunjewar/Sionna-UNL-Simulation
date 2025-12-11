# Sionna-UNL-Simulation

Wireless communication simulation for University of Nebraska-Lincoln buildings using Sionna ray tracing.

## Overview

Simulates WiFi signal propagation across multiple UNL buildings by comparing measured RSSI data with ray-traced predictions. Generates visualizations and validation overlays for building coverage analysis.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- Matplotlib
- Sionna (optional, falls back to simplified model if unavailable)
- TensorFlow (required if using Sionna)

## Installation

```bash
pip install numpy pandas matplotlib
pip install sionna tensorflow  # Optional, for ray tracing
```

## Usage

```bash
python scene_builder.py
python main.py
```

## Files

- `main.py` - Main simulation runner and visualization generator
- `scene_builder.py` - Building scene geometry generator
- `Wireless Communications - Data Collection - Data-4.csv` - Measurement data
- `scenes/` - XML scene files for each building
- `results/` - Generated visualizations and analysis plots

## Output

Generates comparison plots, scene layouts, validation overlays, and summary statistics for each building in the `results/` directory.

## Buildings Supported

Kiewit, Adele Coryell, Love Library South, Selleck, Brace, Hamilton, Bessey, Union, Oldfather, Memorial Stadium, and more.
