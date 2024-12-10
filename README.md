# Handwritten Letters Recognition

Этот проект направлен на распознавание рукописных букв английского алфавита с использованием нейронной сети. Модель обучена на наборе данных, который содержит изображения рукописных букв, и использует полносвязанные слои для классификации.

## Структура проекта

- `data/` - Директория, в которой хранится файл с данными для обучения (`A_Z_Handwritten_Data.csv`).
- `src/` - Исходный код проекта:
  - `model.py` - Содержит функцию для создания модели нейронной сети.
  - `train.py` - Скрипт для обучения модели.
  - `utils.py` - Вспомогательные функции, включая предобработку данных.
  - `visualize.py` - Скрипт для визуализации результатов обучения (графики точности и потерь).
- `history.json` - Файл, содержащий историю обучения модели.
- `handwritten_model.h5` - Сохраненная модель нейронной сети.

# Авторы проекта

[Шайхутдинов Виктор](https://github.com/drugojkira)


## Установка

1. Клонируйте репозиторий:

   ```bash
   git clone https://github.com/ваш_репозиторий.git
   cd ваш_репозиторий

2. Поместите файл (скачанный по ссылке ниже в папку `data/`):

   https://storage.yandexcloud.net/academy.ai/A_Z_Handwritten_Data.csv

3. Активировать виртуальное окружение:

   ```bash
   python -m venv venv
   source venv/bin/activate  # для Linux/macOS
   venv\Scripts\activate     # для Windows

4. Установите необходимые библиотеки:

   ```bash
   pip install -r requirements.txt

5. Для обучения модели запустите скрипт train.py:

   ```bash
   python src/train.py

После выполнения обучения, модель будет сохранена в файл handwritten_model.h5, а история обучения в history.json.

6. Для визуализации графиков точности и потерь, используйте скрипт visualize.py:

   ```bash
   python -c "from src.visualize import plot_training_history; import pandas as pd; history = pd.read_json('history.json'); plot_training_history(history)"


   
