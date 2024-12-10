from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def create_model(num_layers=3, num_neurons=128, input_shape=(128,)):
    """
    Создает полносвязанную нейронную сеть.

    :param num_layers: Число слоев.
    :param num_neurons: Число нейронов в каждом слое.
    :param input_shape: Размер входного слоя.
    :return: Обучаемая модель.
    """
    model = Sequential()
    model.add(Dense(num_neurons, activation='relu', input_shape=input_shape))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation='relu'))
        model.add(Dropout(0.2))
    model.add(Dense(26, activation='softmax'))  # 26 букв в алфавите
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
