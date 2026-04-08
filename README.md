# SSTa ConvLSTM Training Project

This directory contains the Python backend code to train the ConvLSTM model using real NetCDF climate data.

## Prerequisites

Since training a deep learning model on climate data requires significant computational power (preferably a GPU), you should run this code on your local machine, a university server, or a cloud environment like Google Colab.

## Setup Instructions

1. **Download this folder**: Download the `ml-model` folder to your local machine.
2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your data**: Place your NetCDF file (e.g., `historical_sst_data.nc`) in this directory.
5. **Run the training script**:
   ```bash
   python train.py
   ```

## Files Explained

* `train.py`: The main script. It uses `xarray` to read the NetCDF file, handles missing values (land masses), normalizes the data, creates time-series sequences, and trains a TensorFlow/Keras ConvLSTM model.
* `requirements.txt`: Lists all necessary Python packages.

## Output Artifacts

After running `train.py`, the script will generate:
1. `best_sst_convlstm.keras`: The trained TensorFlow model weights.
2. `normalization_stats.npy`: A dictionary containing the `mean` and `std` used to normalize the data. You will need these values during inference to convert the model's output back to real Celsius anomalies.
# sst-train
