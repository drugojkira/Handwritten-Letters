import matplotlib.pyplot as plt


def plot_training_history(history):
    # График точности
    plt.figure(figsize=(12, 6))

    # Точность на обучении и тесте
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Точность на обучении')
    plt.plot(history['val_accuracy'], label='Точность на тестировании')
    plt.title('Точность обучения')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность')
    plt.legend()

    # Потери на обучении и тесте
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Потери на обучении')
    plt.plot(history['val_loss'], label='Потери на тестировании')
    plt.title('Потери обучения')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    plt.tight_layout()
    plt.show()
