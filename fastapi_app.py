import os
import io
import glob
import base64
import numpy as np
import pandas as pd
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Изключваме излишните съобщения от TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ограничаваме TensorFlow да не използва всички CPU ядра, за да не замръзва компютъра
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Глобални променливи за модела и данните
MODEL_PATH = 'best_sst_convlstm.keras' # Сменете с името на вашия модел
DATA_PATH = 'data/*.nc' # Път към вашите .nc файлове
MODEL = None
VAR_NAME = 'sea_surface_temperature_anomaly'

def load_model_if_needed(model_name: str):
    global MODEL, MODEL_PATH
    if MODEL is None or MODEL_PATH != model_name:
        print(f"Зареждане на модел: {model_name}...")
        try:
            MODEL = tf.keras.models.load_model(model_name)
            MODEL_PATH = model_name
            print(f"✅ Моделът {model_name} е зареден успешно!")
            return True
        except Exception as e:
            print(f"❌ Грешка при зареждане на модела {model_name}: {e}")
            return False
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_if_needed(MODEL_PATH)
    yield

@tf.function
def predict_step(model, x):
    return model(x, training=False)

app = FastAPI(
    title="Ocean AI Prediction API",
    description="API за предвиждане на океански данни чрез ConvLSTM модел. Сравнява предвижданията с реални данни, ако са налични.",
    lifespan=lifespan
)

# Разрешаваме достъп от всеки Front-end (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.responses import Response
import tempfile

@app.get("/predict")
async def predict(start_date: str, model: str = 'best_sst_convlstm.keras'):
    """
    Прави предвиждане на базата на 10 дни, започващи от `start_date`.
    Ако ден 11 (target_date) съществува в данните, прави сравнение.
    Формат на датата: YYYY-MM-DD (напр. 2020-01-01)
    """
    if not load_model_if_needed(model):
        raise HTTPException(status_code=500, detail=f"Моделът {model} не може да бъде зареден. Проверете дали файлът съществува.")

    if MODEL is None:
        raise HTTPException(status_code=500, detail="Моделът не е зареден. Проверете сървъра.")

    try:
        # Датата, която потребителят е избрал, е датата, която искаме да предвидим
        target_dt = pd.to_datetime(start_date)
        # Моделът изисква 10 дни входни данни ПРЕДИ тази дата
        start_dt = target_dt - pd.Timedelta(days=10)
        end_dt = target_dt - pd.Timedelta(days=1)
    except Exception:
        raise HTTPException(status_code=400, detail="Невалиден формат на датата. Използвайте YYYY-MM-DD.")

    all_files = sorted(glob.glob(DATA_PATH))
    if not all_files:
        raise HTTPException(status_code=500, detail="Не са намерени .nc файлове в папка data/")

    # Намираме само файловете, които ни трябват (10 дни вход + 1 ден target)
    needed_dates = [start_dt + pd.Timedelta(days=i) for i in range(11)]
    needed_date_strs = [d.strftime('%Y-%m-%d') for d in needed_dates]
    
    file_list = [f for f in all_files if any(d_str in f for d_str in needed_date_strs)]

    if len(file_list) < 10:
        import re
        dates_in_files = sorted([re.search(r'\d{4}-\d{2}-\d{2}', f).group() for f in all_files if re.search(r'\d{4}-\d{2}-\d{2}', f)])
        available_start = dates_in_files[0] if dates_in_files else "N/A"
        available_end = dates_in_files[-1] if dates_in_files else "N/A"
        
        error_msg = f"Не са намерени достатъчно файлове за 10-те дни преди {target_dt.strftime('%Y-%m-%d')} (от {start_dt.strftime('%Y-%m-%d')} до {end_dt.strftime('%Y-%m-%d')}). Намерени: {len(file_list)}. Налични данни: {available_start} до {available_end}"
        print(f"❌ 404 Error: {error_msg}")
        raise HTTPException(status_code=404, detail=error_msg)

    try:
        # Зареждаме само нужните файлове, за да пестим RAM и време
        with xr.open_mfdataset(file_list, combine='nested', concat_dim='time', engine='netcdf4') as ds:
            var_name = VAR_NAME
            if var_name not in ds.data_vars:
                var_name = list(ds.data_vars)[0]

            # 1. Извличане на 10-те дни за вход
            input_ds = ds.isel(time=slice(0, 10))
            
            if len(input_ds.time) < 10:
                import re
                dates_in_files = sorted([re.search(r'\d{4}-\d{2}-\d{2}', f).group() for f in all_files if re.search(r'\d{4}-\d{2}-\d{2}', f)])
                available_start = dates_in_files[0] if dates_in_files else "N/A"
                available_end = dates_in_files[-1] if dates_in_files else "N/A"
                
                error_msg = f"Намерени са само {len(input_ds.time)} времеви стъпки в тези файлове. Трябват точно 10. Налични данни: {available_start} до {available_end}"
                print(f"❌ 404 Error: {error_msg}")
                raise HTTPException(
                    status_code=404, 
                    detail=error_msg
                )

            data_array = input_ds[var_name].values
            
            # Extract min and max temperatures for each time step before replacing NaNs with 0
            input_min_temps = [float(np.nanmin(data_array[i])) for i in range(data_array.shape[0])]
            input_max_temps = [float(np.nanmax(data_array[i])) for i in range(data_array.shape[0])]
            
            # Create land mask from the first frame
            land_mask = np.isnan(data_array[0]) # True for land
            land_mask_tensor = tf.expand_dims(tf.expand_dims(land_mask.astype(np.float32), axis=-1), axis=0)
            land_mask_resized = tf.image.resize(land_mask_tensor, [511, 1080], method='nearest')
            land_mask_2d = np.squeeze(land_mask_resized.numpy()) > 0.5
            
            data_array = np.nan_to_num(data_array, nan=0.0)

            # 2. Преоразмеряване до (511, 1080)
            input_tensor = tf.expand_dims(data_array, axis=-1) # (10, H, W, 1)
            resized_frames = tf.image.resize(input_tensor, [511, 1080]) # (10, 511, 1080, 1)
            model_input = tf.expand_dims(resized_frames, axis=0) # (1, 10, 511, 1080, 1)
            
            resized_frames_np = np.squeeze(resized_frames.numpy())
            for i in range(10):
                resized_frames_np[i][land_mask_2d] = -1000.0

            # 3. Предвиждане
            prediction = predict_step(MODEL, model_input).numpy()
            pred_2d = np.squeeze(prediction)
            pred_2d[land_mask_2d] = -1000.0
            
            # Extract min and max temperatures for prediction data
            valid_pred = pred_2d[~land_mask_2d]
            pred_min_temp = float(np.nanmin(valid_pred)) if len(valid_pred) > 0 else 0.0
            pred_max_temp = float(np.nanmax(valid_pred)) if len(valid_pred) > 0 else 0.0

            # 4. Генериране на изходен Dataset
            out_vars = {
                'input': (['time', 'lat', 'lon'], resized_frames_np.astype(np.float32)),
                'prediction': (['lat', 'lon'], pred_2d.astype(np.float32)),
            }

            mse_val = -1.0
            msg = f"Предвиждането за {target_dt.strftime('%Y-%m-%d')} е успешно. Няма реални данни за сравнение за тази дата."

            actual_min_temp = -999.0
            actual_max_temp = 999.0
            if len(ds.time) > 10:
                actual_ds = ds.isel(time=slice(10, 11))
                actual_array = actual_ds[var_name].values[0]
                
                # Extract min and max temperatures for actual data
                actual_min_temp = float(np.nanmin(actual_array))
                actual_max_temp = float(np.nanmax(actual_array))
                
                actual_array = np.nan_to_num(actual_array, nan=0.0)
                
                # Преоразмеряваме и реалните данни, за да съвпадат с предвиждането
                actual_tensor = tf.expand_dims(tf.expand_dims(actual_array, axis=-1), axis=0)
                actual_resized = tf.image.resize(actual_tensor, [511, 1080])
                actual_2d = np.squeeze(actual_resized.numpy())
                actual_2d[land_mask_2d] = -1000.0
                
                # Изчисляване на разликата (Грешката)
                difference = pred_2d - actual_2d
                difference[land_mask_2d] = -1000.0
                
                valid_diff = difference[~land_mask_2d]
                mse_val = float(np.mean(np.square(valid_diff))) if len(valid_diff) > 0 else 0.0
                
                out_vars['actual'] = (['lat', 'lon'], actual_2d.astype(np.float32))
                out_vars['difference'] = (['lat', 'lon'], difference.astype(np.float32))
                
                msg = f"Data found for: {target_dt.strftime('%Y-%m-%d')}"

            ds_out = xr.Dataset(out_vars)
            ds_out.attrs['message'] = msg
            ds_out.attrs['mse'] = mse_val
            ds_out.attrs['input_min_temps'] = input_min_temps
            ds_out.attrs['input_max_temps'] = input_max_temps
            ds_out.attrs['pred_min_temp'] = pred_min_temp
            ds_out.attrs['pred_max_temp'] = pred_max_temp
            if len(ds.time) > 10:
                ds_out.attrs['actual_min_temp'] = actual_min_temp
                ds_out.attrs['actual_max_temp'] = actual_max_temp
            ds_out.attrs['start_date'] = start_dt.strftime('%Y-%m-%d')
            ds_out.attrs['target_date'] = target_dt.strftime('%Y-%m-%d')
            ds_out.attrs['has_actual'] = 1 if len(ds.time) > 10 else 0

            with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
                tmp_name = tmp.name
                
            ds_out.to_netcdf(tmp_name, format='NETCDF3_CLASSIC', engine='netcdf4')
            
            with open(tmp_name, 'rb') as f:
                nc_bytes = f.read()
            os.remove(tmp_name)
            
            # Изчистваме кеша на xarray за да предотвратим memory leaks
            xr.backends.file_manager.FILE_CACHE.clear()
            
            import gc
            gc.collect()

            return Response(content=nc_bytes, media_type="application/x-netcdf")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Грешка при обработка на данните: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)