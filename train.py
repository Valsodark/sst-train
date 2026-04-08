import os
import glob
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import Sequence

class SSTDataGenerator(Sequence):
    """
    Generates batches of data on the fly. 
    This prevents Out-Of-Memory (OOM) errors in System RAM (60GB) by not 
    duplicating the overlapping sequences in memory.
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
            
        X = np.array(X)
        y = np.array(y)
        
        # Add channel dimension (1 channel for SST)
        X = X[..., np.newaxis]
        y = y[..., np.newaxis]
        
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def load_and_preprocess_data(data_dir, variable_name='sst', downsample_factor=4):
    """
    Loads NetCDF files, downsamples them to fit in VRAM, and normalizes them.
    """
    file_pattern = os.path.join(data_dir, '*.nc')
    files = sorted(glob.glob(file_pattern))
    
    if len(files) == 0:
        print(f"⚠️ No NetCDF files found in {data_dir}. Generating dummy data.")
        # Dummy data: 100 days, downsampled grid
        data = np.random.rand(100, 2042 // downsample_factor, 4320 // downsample_factor).astype(np.float32)
        return data, 0.5, 0.2
        
    print(f"Loading {len(files)} NetCDF files from {data_dir}...")
    print(f"Applying {downsample_factor}x spatial downsampling to fit in 32GB VRAM...")
    
    data_list = []
    actual_var_name = variable_name
    
    for i, f in enumerate(files):
        try:
            ds = xr.open_dataset(f)
            
            # Auto-detect variable name
            if i == 0:
                if variable_name not in ds.data_vars:
                    possible_vars = [v for v in ds.data_vars if 'bnds' not in v and 'bounds' not in v]
                    if possible_vars:
                        actual_var_name = possible_vars[0]
                        print(f"⚠️ Variable '{variable_name}' not found. Using '{actual_var_name}'.")
                    else:
                        raise ValueError(f"No suitable data variables found in {f}")
            
            val = ds[actual_var_name].values
            
            # Extract 2D grid
            if val.ndim == 4:
                val = val[0, 0]
            elif val.ndim == 3:
                val = val[0]
            elif val.ndim > 4:
                while val.ndim > 2:
                    val = val[0]
            
            if val.ndim == 2:
                # DOWNSAMPLING: Take every Nth pixel to reduce memory by N^2
                # e.g., factor=4 reduces memory footprint by 16x
                val = val[::downsample_factor, ::downsample_factor]
                data_list.append(val)
            else:
                print(f"⚠️ Unexpected shape {val.shape} in {f}. Skipping.")
                
            ds.close()
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")
            
    if not data_list:
        raise ValueError("No valid data could be loaded.")
        
    # Convert to single numpy array
    data = np.array(data_list, dtype=np.float32)
    print(f"Loaded and downsampled data shape: {data.shape}")
    
    # Handle NaNs (Land masses)
    data = np.nan_to_num(data, nan=0.0)

    # Normalization
    mean = np.mean(data)
    std = np.std(data)
    data = (data - mean) / (std + 1e-7)
    print(f"Data normalized. Mean: {mean:.4f}, Std: {std:.4f}")

    return data, mean, std

def build_convlstm_model(input_shape):
    """
    Builds a Convolutional LSTM model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.ConvLSTM2D(
            filters=32, 
            kernel_size=(3, 3), 
            padding='same', 
            return_sequences=True,
            activation='tanh'
        ),
        layers.BatchNormalization(),
        
        layers.ConvLSTM2D(
            filters=32, 
            kernel_size=(3, 3), 
            padding='same', 
            return_sequences=False,
            activation='tanh'
        ),
        layers.BatchNormalization(),
        
        layers.Conv2D(
            filters=1, 
            kernel_size=(3, 3), 
            activation='linear', 
            padding='same'
        )
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

if __name__ == "__main__":
    # --- Configuration for RTX 5090 (32GB VRAM) & 60GB RAM ---
    DATA_DIR = './data'
    VARIABLE_NAME = 'analysed_sst'
    SEQ_LENGTH = 10
    
    # DOWNSAMPLE_FACTOR: 4 means resolution is reduced by 4x (e.g. 4000x2000 -> 1000x500)
    # This reduces VRAM usage by 16x. Essential for global high-res data.
    DOWNSAMPLE_FACTOR = 4 
    
    # BATCH_SIZE: 4 is safe for 32GB VRAM with downsampled data. 
    # If it OOMs, reduce to 2. If GPU utilization is low, increase to 8.
    BATCH_SIZE = 4
    EPOCHS = 20

    # 1. Prepare Data (Loads into 60GB RAM efficiently)
    data, data_mean, data_std = load_and_preprocess_data(
        DATA_DIR, 
        variable_name=VARIABLE_NAME, 
        downsample_factor=DOWNSAMPLE_FACTOR
    )
    
    # Split into training and validation sets (80% train, 20% val) sequentially
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Training frames: {len(train_data)}")
    print(f"Validation frames: {len(val_data)}")

    # 2. Create Data Generators (Saves System RAM)
    train_gen = SSTDataGenerator(train_data, SEQ_LENGTH, BATCH_SIZE, shuffle=True)
    val_gen = SSTDataGenerator(val_data, SEQ_LENGTH, BATCH_SIZE, shuffle=False)

    # 3. Build Model
    # input_shape = (time_steps, lat, lon, channels)
    input_shape = (SEQ_LENGTH, data.shape[1], data.shape[2], 1)
    model = build_convlstm_model(input_shape)
    model.summary()

    # 4. Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_sst_convlstm.keras', 
            monitor='val_loss',
            save_best_only=True
        )
    ]

    # 5. Train Model
    print("\\nStarting training on RTX 5090...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        workers=8, # Use 8 out of 12 vCPUs for faster data loading
        use_multiprocessing=False
    )

    # 6. Save final artifacts
    model.save('final_sst_convlstm.keras')
    np.save('normalization_stats.npy', {'mean': data_mean, 'std': data_std})
    
    print("\\n✅ Training complete. Model and normalization stats saved.")
