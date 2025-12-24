"""
Конфигурационные параметры проекта
"""
import os
import torch

# ============================================
# НАСТРОЙКИ GPU
# ============================================
class GPUConfig:
    USE_GPU = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if USE_GPU else 'cpu')
    GPU_ID = 0 if USE_GPU else None
    
    @staticmethod
    def print_gpu_info():
        if GPUConfig.USE_GPU:
            print(f"🎮 GPU доступна: {torch.cuda.get_device_name(0)}")
            print(f"🎮 Память GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("⚠️ GPU не доступна, используется CPU")

# ============================================
# ПАТИ ДАННЫХ (адаптировано под вашу структуру)
# ============================================
class DataConfig:
    # Базовый путь - проверьте, что папка data существует
    BASE_DIR = './data'
    
    # ПУТИ ДЛЯ ТРЕНИРОВОЧНЫХ ДАННЫХ
    TRAIN_FAKE_PATHS = [
        os.path.join(BASE_DIR, 'overall', 'overall', 'celebrityDataset', 'fake'),
        os.path.join(BASE_DIR, 'training', 'training', 'fakeNewsDataset', 'fake'),
        os.path.join(BASE_DIR, 'overall', 'overall', 'fakeNewsDataset', 'fake')
    ]
    
    TRAIN_REAL_PATHS = [
        os.path.join(BASE_DIR, 'overall', 'overall', 'celebrityDataset', 'legit'),
        os.path.join(BASE_DIR, 'training', 'training', 'fakeNewsDataset', 'real'),
        os.path.join(BASE_DIR, 'overall', 'overall', 'fakeNewsDataset', 'real')
    ]
    
    # ПУТИ ДЛЯ ТЕСТОВЫХ ДАННЫХ
    TEST_FAKE_PATH = os.path.join(BASE_DIR, 'Testing_dataset', 'testingSet', 'fake')
    TEST_REAL_PATH = os.path.join(BASE_DIR, 'Testing_dataset', 'testingSet', 'real')
    
    # Минимальная частота слова для включения в словарь
    MIN_WORD_COUNT = 3

# ============================================
# НАСТРОЙКИ ПРОГРЕСС-БАРОВ
# ============================================
class ProgressConfig:
    ENABLE_PROGRESS_BARS = True
    PROGRESS_BAR_COLOR = 'green'
    BAR_FORMAT = '{l_bar}{bar:30}{r_bar}{bar:-30b}'
    UPDATE_INTERVAL = 0.1  # секунды

# ============================================
# ОСТАЛЬНЫЕ КОНФИГУРАЦИИ
# ============================================
# В config.py УВЕЛИЧЬТЕ эти значения:
class Word2VecConfig:
    EMBEDDING_DIM = 200  # было 100
    WINDOW_SIZE = 5      # было 3  
    LEARNING_RATE = 0.025  # было 0.1 (меньше для стабильности)
    EPOCHS = 20         # было 5
    BATCH_SIZE = 1000

# В config.py:
class NeuralNetConfig:
    HIDDEN_LAYERS = [256, 128, 64]  # было [64, 32]
    OUTPUT_DIM = 1
    LEARNING_RATE = 0.001  # было 0.01 (меньше для лучшей сходимости)
    EPOCHS = 100           # было 50
    BATCH_SIZE = 64        # было 32
    VALIDATION_SPLIT = 0.2

class PreprocessingConfig:
    MIN_WORD_LENGTH = 3
    REMOVE_STOPWORDS = True
    LEMMATIZE = False

class OutputConfig:
    BASE_DIR = 'results'
    GRAPHS_DIR = os.path.join(BASE_DIR, 'graphs')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    RESULTS_FILE = os.path.join(REPORTS_DIR, 'neural_network_results.txt')
    METRICS_FILE = os.path.join(REPORTS_DIR, 'model_metrics.json')
    GRAPH_NAMES = {
        'learning_curve': 'learning_curve.png',
        'validation_accuracy': 'validation_accuracy.png',
        'prediction_distribution': 'prediction_distribution.png',
        'important_words': 'important_words.png',
        'confusion_matrix': 'confusion_matrix.png',
        'class_specific_words': 'class_specific_words.png'
    }

class LoggingConfig:
    VERBOSE = True
    PRINT_EVERY_EPOCH = 10
    SAVE_INTERMEDIATE = True