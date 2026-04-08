import os
import glob
import numpy as np
import xarray as xr
import tensorflow as tf
import matplotlib.pyplot as plt

# Изключваме излишните съобщения от TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():
    print("1. Зареждане на модела...")
    model_path = 'best_sst_convlstm.keras' # Сменете с името на вашия модел
    model = tf.keras.models.load_model(model_path)
    print("Моделът е зареден!")

    print("2. Търсене на 10 .nc файла...")
    # ТУК: Сменете 'data/' с папката, в която са вашите .nc файлове
    # Взимаме първите 10 файла, подредени по азбучен ред/дата
    file_list = sorted(glob.glob('data/*.nc'))[:10] 
    
    if len(file_list) < 10:
        print(f"ГРЕШКА: Намерени са само {len(file_list)} файла. Трябват 10!")
        return
    
    print(f"Зареждане на следните файлове:\n{file_list}")

    # Зареждаме и обединяваме 10-те файла по време (time)
    ds = xr.open_mfdataset(file_list, combine='nested', concat_dim='time')
    
    # ТУК: Напишете името на променливата във вашите .nc файлове 
    # (например 'thetao' за температура, 'so' за соленост, 'zos' за морско ниво)
    var_name = 'sea_surface_temperature_anomaly' 
    
    # Извличаме данните като numpy масив. Очакван формат: (10, височина, ширина)
    data_array = ds[var_name].values 
    
    # Почистване на липсващи данни (NaN), ако има такива (често срещано при океаните)
    data_array = np.nan_to_num(data_array, nan=0.0)

    print(f"Формат на извлечените данни: {data_array.shape}")

    # 3. Подготовка на данните за модела
    # Повечето модели очакват формат: (Batch, Time, Height, Width, Channels)
    # Добавяме измерение за Batch (отпред) и Channels (отзад)
    input_data = np.expand_dims(data_array, axis=0)  # Става (1, 10, H, W)
    input_data = np.expand_dims(input_data, axis=-1) # Става (1, 10, H, W, 1)
    
    print(f"Оригинален формат: {input_data.shape}")

    # --- НОВО: Преоразмеряване на данните ---
    # Моделът е трениран на по-малки изображения (511x1080), за да не дава OOM грешка.
    # Затова трябва да смалим и тези данни до същия размер преди предвиждането.
    print("Преоразмеряване на данните до (511, 1080)...")
    frames = input_data[0] # Взимаме 10-те кадъра: формат (10, 2041, 4320, 1)
    resized_frames = tf.image.resize(frames, [511, 1080]) # Преоразмеряваме с TensorFlow
    input_data = np.expand_dims(resized_frames.numpy(), axis=0) # Връщаме в (1, 10, 511, 1080, 1)
    
    print(f"Формат, подаден към модела: {input_data.shape}")

    # 4. ПРАВИМ ПРЕДВИЖДАНЕТО ЗА СЛЕДВАЩИЯ ДЕН
    print("3. Генериране на предвиждане за следващия ден...")
    prediction = model.predict(input_data)
    
    # Махаме излишните измерения, за да остане само 2D картата (Height, Width)
    # Ако моделът връща (1, 1, H, W, 1), това ще извади само H и W
    pred_2d = np.squeeze(prediction) 
    
    print(f"Формат на предвиждането: {pred_2d.shape}")

    # 5. ВИЗУАЛИЗАЦИЯ С MATPLOTLIB
    print("4. Рисуване на картата...")
    plt.figure(figsize=(10, 8))
    
    # Рисуваме данните. cmap='jet' или 'viridis' са добри за температурни/океански карти
    plt.imshow(pred_2d, cmap='viridis', origin='lower') 
    
    plt.colorbar(label=f'Predicted {var_name}')
    plt.title("AI Prediction for the Next Day")
    plt.xlabel("Longitude (Pixels)")
    plt.ylabel("Latitude (Pixels)")
    
    # ЗАПАЗВАМЕ КАТО СНИМКА (тъй като сме на сървър без монитор)
    output_image = "prediction_next_day.png"
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"✅ ГОТОВО! Предвиждането е запазено като снимка: {output_image}")

if __name__ == "__main__":
    main()