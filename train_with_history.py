"""
train.py - SST ConvLSTM training with history logging and baseline evaluation.

Drop-in replacement for the original train.py. Same model and hyperparameters,
but additionally:
  - Saves training history to training_history.json
  - Generates training_curves.png (MSE + MAE, train + val)
  - Computes and saves final validation metrics
  - Computes persistence baseline (tomorrow = today) for comparison
  - Saves all metrics to evaluation_results.json
"""

import os
import glob
import json
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless RunPod
import matplotlib.pyplot as plt


class SSTDataGenerator(Sequence):
    """
    Generates batches of data on the fly to prevent OOM errors.
    """
    def __init__(self, data, seq_length, batch_size, shuffle=True):
        self.data = data
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.data) - self.seq_length)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        X, y = [], []
        for i in batch_indices:
            X.append(self.data[i : i + self.seq_length])
            y.append(self.data[i + self.seq_length])
        X = np.array(X)[..., np.newaxis]
        y = np.array(y)[..., np.newaxis]
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def load_and_preprocess_data(data_dir, variable_name='analysed_sst', downsample_factor=4):
    """Load NetCDF files, downsample, handle NaNs, normalize."""
    file_pattern = os.path.join(data_dir, '*.nc')
    files = sorted(glob.glob(file_pattern))

    if len(files) == 0:
        raise ValueError(f"No NetCDF files found in {data_dir}")

    print(f"Loading {len(files)} NetCDF files from {data_dir}...")
    print(f"Applying {downsample_factor}x spatial downsampling...")

    data_list = []
    actual_var_name = variable_name

    for i, f in enumerate(files):
        try:
            ds = xr.open_dataset(f)
            if i == 0:
                if variable_name not in ds.data_vars:
                    possible_vars = [v for v in ds.data_vars if 'bnds' not in v and 'bounds' not in v]
                    if possible_vars:
                        actual_var_name = possible_vars[0]
                        print(f"Variable '{variable_name}' not found. Using '{actual_var_name}'.")

            val = ds[actual_var_name].values
            if val.ndim == 4:
                val = val[0, 0]
            elif val.ndim == 3:
                val = val[0]
            elif val.ndim > 4:
                while val.ndim > 2:
                    val = val[0]

            if val.ndim == 2:
                val = val[::downsample_factor, ::downsample_factor]
                data_list.append(val)
            ds.close()
        except Exception as e:
            print(f"Error loading {f}: {e}")

    if not data_list:
        raise ValueError("No valid data could be loaded.")

    data = np.array(data_list, dtype=np.float32)
    print(f"Loaded data shape: {data.shape}")

    # Handle NaNs (land)
    data = np.nan_to_num(data, nan=0.0)

    # Z-score normalization
    mean = float(np.mean(data))
    std = float(np.std(data))
    data = (data - mean) / (std + 1e-7)
    print(f"Data normalized. Mean: {mean:.4f}, Std: {std:.4f}")

    return data, mean, std


def build_convlstm_model(input_shape):
    """ConvLSTM architecture: 2x ConvLSTM2D + BN, Conv2D output."""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',
                          return_sequences=True, activation='tanh'),
        layers.BatchNormalization(),
        layers.ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same',
                          return_sequences=False, activation='tanh'),
        layers.BatchNormalization(),
        layers.Conv2D(filters=1, kernel_size=(3, 3), activation='linear', padding='same'),
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def plot_training_curves(history, output_path='training_curves.png'):
    """Generate the loss curve figure for the thesis (Фиг. 3.2)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    epochs_range = range(1, len(history.history['loss']) + 1)

    # MSE plot
    axes[0].plot(epochs_range, history.history['loss'], 'b-', label='Обучаващ набор', linewidth=2)
    axes[0].plot(epochs_range, history.history['val_loss'], 'r-', label='Валидационен набор', linewidth=2)
    axes[0].set_xlabel('Епоха', fontsize=12)
    axes[0].set_ylabel('MSE загуба', fontsize=12)
    axes[0].set_title('Средноквадратична грешка', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Mark the best epoch
    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.6,
                    label=f'Най-добра епоха: {best_epoch}')
    axes[0].legend(fontsize=11)

    # MAE plot
    axes[1].plot(epochs_range, history.history['mae'], 'b-', label='Обучаващ набор', linewidth=2)
    axes[1].plot(epochs_range, history.history['val_mae'], 'r-', label='Валидационен набор', linewidth=2)
    axes[1].set_xlabel('Епоха', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Средна абсолютна грешка', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {output_path}")


def compute_persistence_baseline(val_data, seq_length):
    """
    Persistence baseline: predict y_{t+1} = y_t (i.e. tomorrow = today).
    Computes MSE and MAE on the same validation samples the model sees.
    """
    mse_list, mae_list = [], []
    for i in range(len(val_data) - seq_length):
        last_input_day = val_data[i + seq_length - 1]  # last day of input sequence
        actual_next = val_data[i + seq_length]         # the day model is asked to predict
        diff = last_input_day - actual_next
        mse_list.append(float(np.mean(diff ** 2)))
        mae_list.append(float(np.mean(np.abs(diff))))
    return float(np.mean(mse_list)), float(np.mean(mae_list))


if __name__ == "__main__":
    # --- Configuration for RTX PRO 4500 Blackwell (32GB VRAM) ---
    DATA_DIR = './data'
    VARIABLE_NAME = 'analysed_sst'
    SEQ_LENGTH = 10
    DOWNSAMPLE_FACTOR = 4
    BATCH_SIZE = 4
    EPOCHS = 20

    # 1. Load and preprocess data
    data, data_mean, data_std = load_and_preprocess_data(
        DATA_DIR, variable_name=VARIABLE_NAME, downsample_factor=DOWNSAMPLE_FACTOR
    )

    # 2. Sequential 80/20 split (do NOT shuffle for time series)
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"Training frames: {len(train_data)}")
    print(f"Validation frames: {len(val_data)}")

    # 3. Generators
    train_gen = SSTDataGenerator(train_data, SEQ_LENGTH, BATCH_SIZE, shuffle=True)
    val_gen = SSTDataGenerator(val_data, SEQ_LENGTH, BATCH_SIZE, shuffle=False)

    # 4. Model
    input_shape = (SEQ_LENGTH, data.shape[1], data.shape[2], 1)
    model = build_convlstm_model(input_shape)
    model.summary()

    # 5. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_sst_convlstm.keras', monitor='val_loss', save_best_only=True
        ),
        tf.keras.callbacks.CSVLogger('training_log.csv'),  # backup of history
    ]

    # 6. Train
    print("\nStarting training on RTX PRO 4500 Blackwell...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        workers=8,
        use_multiprocessing=False,
    )

    # 7. Save model and stats
    model.save('final_sst_convlstm.keras')
    np.save('normalization_stats.npy', {'mean': data_mean, 'std': data_std})

    # 8. Save full training history
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open('training_history.json', 'w') as f:
        json.dump(history_serializable, f, indent=2)
    print("Training history saved to training_history.json")

    # 9. Plot training curves (Фиг. 3.2 for the thesis)
    plot_training_curves(history, 'training_curves.png')

    # 10. Final evaluation on validation set
    print("\nEvaluating final model on validation set...")
    final_val_loss, final_val_mae = model.evaluate(val_gen, verbose=1)

    # 11. Persistence baseline (tomorrow = today)
    print("\nComputing persistence baseline (tomorrow = today)...")
    baseline_mse, baseline_mae = compute_persistence_baseline(val_data, SEQ_LENGTH)

    # 12. Compute improvement
    mse_improvement_pct = (baseline_mse - final_val_loss) / baseline_mse * 100
    mae_improvement_pct = (baseline_mae - final_val_mae) / baseline_mae * 100

    # 13. Save and print results
    results = {
        'hardware': 'NVIDIA RTX PRO 4500 Blackwell, 32 GB VRAM',
        'seq_length': SEQ_LENGTH,
        'batch_size': BATCH_SIZE,
        'epochs_configured': EPOCHS,
        'epochs_run': len(history.history['loss']),
        'best_epoch': int(np.argmin(history.history['val_loss'])) + 1,
        'downsample_factor': DOWNSAMPLE_FACTOR,
        'train_frames': len(train_data),
        'val_frames': len(val_data),
        'data_mean_celsius': data_mean,
        'data_std_celsius': data_std,
        'convlstm_final_val_mse': float(final_val_loss),
        'convlstm_final_val_mae': float(final_val_mae),
        'persistence_baseline_mse': baseline_mse,
        'persistence_baseline_mae': baseline_mae,
        'mse_improvement_over_baseline_pct': float(mse_improvement_pct),
        'mae_improvement_over_baseline_pct': float(mae_improvement_pct),
    }
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("FINAL RESULTS (for thesis)")
    print("=" * 60)
    print(f"ConvLSTM       MSE: {final_val_loss:.6f}   MAE: {final_val_mae:.6f}")
    print(f"Persistence    MSE: {baseline_mse:.6f}   MAE: {baseline_mae:.6f}")
    print(f"Improvement    MSE: {mse_improvement_pct:+.2f}%   MAE: {mae_improvement_pct:+.2f}%")
    print("=" * 60)
    print("\nFiles produced:")
    print("  - best_sst_convlstm.keras       (best model by val_loss)")
    print("  - final_sst_convlstm.keras      (model at end of training)")
    print("  - normalization_stats.npy       (mean and std for inference)")
    print("  - training_history.json         (per-epoch loss/mae values)")
    print("  - training_log.csv              (same, CSV backup)")
    print("  - training_curves.png           (Фиг. 3.2 for thesis)")
    print("  - evaluation_results.json       (final metrics + baseline)")
