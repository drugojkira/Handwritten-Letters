import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from model import create_model
from utils import preprocess_data

# Добавляем текущую директорию в PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Путь к данным
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "A_Z_Handwritten_Data.csv")

# Загрузка данных
data = pd.read_csv(data_path)
X, y = preprocess_data(data)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Создание модели
model = create_model(num_layers=3, num_neurons=128,
                     input_shape=(X_train.shape[1],))

# Обучение модели
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10,
                    batch_size=128)

# Сохранение модели и истории
model.save(os.path.join(current_dir, "..", "handwritten_model.h5"))
pd.DataFrame(history.history).to_json(os.path.join(current_dir, "..",
                                                   "history.json"))
