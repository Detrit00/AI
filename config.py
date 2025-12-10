import torch
import os

#пути к данным и результатам
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "stanford_dogs", "Images")     #путь к папке с датасетом собак
RESULTS_PATH = os.path.join(BASE_DIR, "results")       #папка для сохранения результатов

# Параметры обучения
BATCH_SIZE = 32                             #количество изображений в одной порции (batch)
NUM_EPOCHS = 10                             #количество полных проходов через весь датасет
LEARNING_RATE = 0.0001                      #скорость обучения (шаг оптимизатора)
NUM_CLASSES = 120                           #количество пород собак в датасете

#настройки устройства (GPU/CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#автоматическое определение: использует GPU если доступен, иначе CPU

print(f"Используется устройство: {DEVICE}")
#вывод информации о используемом устройстве для отладки