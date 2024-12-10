import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical


def preprocess_data(data):
    """
    Предобрабатывает данные для обучения.

    :param data: DataFrame с исходными данными.
    :return: Кортеж (X, y) с подготовленными признаками и метками.
    """
    X = data.iloc[:, 1:].values.astype('float32') / 255.0  # Нормализация
    y = data.iloc[:, 0].values  # Метки классов
    y = to_categorical(y, num_classes=26)  # One-hot encoding
    return X, y
