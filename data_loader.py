"""
Модуль для загрузки и предобработки данных с прогресс-барами
"""
import os
import re
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

class DataLoader:
    def __init__(self, config, progress_config):
        self.config = config
        self.progress_config = progress_config
    
    def load_texts_from_folder(self, folder_path, label, desc="Загрузка файлов"):
        """Загрузка всех текстовых файлов из папки с заданной меткой."""
        texts, labels = [], []
        
        if not os.path.exists(folder_path):
            print(f"Предупреждение: путь {folder_path} не существует")
            return texts, labels
        
        # Получаем список всех текстовых файлов
        all_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    all_files.append(os.path.join(root, file))
        
        if not all_files:
            return texts, labels
        
        # Используем tqdm для отображения прогресса
        if self.progress_config.ENABLE_PROGRESS_BARS:
            file_iterator = tqdm(
                all_files,
                desc=desc,
                bar_format=self.progress_config.BAR_FORMAT,
                colour=self.progress_config.PROGRESS_BAR_COLOR
            )
        else:
            file_iterator = all_files
        
        for file_path in file_iterator:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    if text:
                        texts.append(text)
                        labels.append(label)
            except Exception as e:
                if not self.progress_config.ENABLE_PROGRESS_BARS:
                    print(f"Ошибка при чтении файла {file_path}: {e}")
                continue
        
        return texts, labels
    
    def load_all_data(self):
        """Загрузка всех тренировочных и тестовых данных с фильтрацией путей."""
        print("="*60)
        print("ЗАГРУЗКА ДАННЫХ")
        print("="*60)
        
        # Функция для фильтрации существующих путей
        def filter_existing_paths(paths):
            return [p for p in paths if os.path.exists(p)]
        
        # Фильтруем существующие пути
        existing_fake_paths = filter_existing_paths(self.config.TRAIN_FAKE_PATHS)
        existing_real_paths = filter_existing_paths(self.config.TRAIN_REAL_PATHS)
        
        print(f"Найдено существующих путей для фейковых новостей: {len(existing_fake_paths)}")
        print(f"Найдено существующих путей для реальных новостей: {len(existing_real_paths)}")
        
        # Загрузка тренировочных данных с прогресс-барами
        train_fake_texts, train_fake_labels = [], []
        train_real_texts, train_real_labels = [], []
        
        for i, path in enumerate(existing_fake_paths):
            texts, labels = self.load_texts_from_folder(
                path, 1, 
                desc=f"Фейковые новости [{i+1}/{len(existing_fake_paths)}]"
            )
            train_fake_texts.extend(texts)
            train_fake_labels.extend(labels)
        
        for i, path in enumerate(existing_real_paths):
            texts, labels = self.load_texts_from_folder(
                path, 0,
                desc=f"Реальные новости [{i+1}/{len(existing_real_paths)}]"
            )
            train_real_texts.extend(texts)
            train_real_labels.extend(labels)
        
        # Загрузка тестовых данных
        if os.path.exists(self.config.TEST_FAKE_PATH):
            test_fake_texts, test_fake_labels = self.load_texts_from_folder(
                self.config.TEST_FAKE_PATH, 1, "Тестовые фейковые новости"
            )
        else:
            test_fake_texts, test_fake_labels = [], []
            print(f"Предупреждение: путь для тестовых фейковых новостей не существует")
        
        if os.path.exists(self.config.TEST_REAL_PATH):
            test_real_texts, test_real_labels = self.load_texts_from_folder(
                self.config.TEST_REAL_PATH, 0, "Тестовые реальные новости"
            )
        else:
            test_real_texts, test_real_labels = [], []
            print(f"Предупреждение: путь для тестовых реальных новостей не существует")
        
        # Создание DataFrame
        train_df = pd.DataFrame({
            'text': train_fake_texts + train_real_texts,
            'label': train_fake_labels + train_real_labels
        })
        
        test_df = pd.DataFrame({
            'text': test_fake_texts + test_real_texts,
            'label': test_fake_labels + test_real_labels
        })
        
        print(f"\n{'='*40}")
        print("Итоги загрузки данных:")
        print(f"Размер тренировочного датасета: {train_df.shape}")
        print(f"Размер тестового датасета: {test_df.shape}")
        
        if len(train_df) > 0:
            print(f"Распределение классов в тренировочных данных:")
            print(train_df['label'].value_counts())
        else:
            print("ВНИМАНИЕ: Тренировочный датасет пуст!")
        
        return train_df, test_df
    
    def simple_preprocess_text(self, text):
        """Улучшенная предобработка текста."""
        # Приведите к нижнему регистру
        text = text.lower()
        
        # Удалите HTML-теги
        text = re.sub(r'<.*?>', '', text)
        
        # Удалите URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Удалите специальные символы, но сохраните знаки препинания для контекста
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Удалите числа
        text = re.sub(r'\d+', '', text)
        
        # Удалите лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Разделите на слова
        tokens = text.split()
        
        # Фильтр по длине и удаление стоп-слов
        basic_stopwords = {'the', 'and', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        # Используем MIN_WORD_COUNT вместо MIN_WORD_LENGTH (MIN_WORD_COUNT уже есть в DataConfig)
        min_word_length = 3  # Задаем минимальную длину слова
        tokens = [token for token in tokens 
                  if len(token) >= min_word_length  # Фиксированное значение
                  and token not in basic_stopwords]
        
        return tokens
    
    def preprocess_dataframes(self, train_df, test_df):
        """Применение предобработки к данным с прогресс-баром."""
        print("\n" + "="*60)
        print("ПРЕДОБРАБОТКА ТЕКСТОВ")
        print("="*60)
        
        # Обработка тренировочных данных
        if self.progress_config.ENABLE_PROGRESS_BARS:
            tqdm.pandas(desc="Обработка тренировочных текстов", bar_format=self.progress_config.BAR_FORMAT)
            train_df['tokens'] = train_df['text'].progress_apply(self.simple_preprocess_text)
            
            tqdm.pandas(desc="Обработка тестовых текстов", bar_format=self.progress_config.BAR_FORMAT)
            test_df['tokens'] = test_df['text'].progress_apply(self.simple_preprocess_text)
        else:
            train_df['tokens'] = train_df['text'].apply(self.simple_preprocess_text)
            test_df['tokens'] = test_df['text'].apply(self.simple_preprocess_text)
        
        # Удаление пустых текстов
        train_df = train_df[train_df['tokens'].apply(len) > 0]
        test_df = test_df[test_df['tokens'].apply(len) > 0]
        
        print(f"\nПосле обработки:")
        print(f"Тренировочный датасет: {train_df.shape}")
        print(f"Тестовый датасет: {test_df.shape}")
        
        return train_df, test_df
    
    def build_vocabulary(self, train_df):
        """Построение словаря на основе тренировочных данных."""
        print("\n" + "="*60)
        print("ПОСТРОЕНИЕ СЛОВАРЯ")
        print("="*60)
        
        # Собираем все токены с прогресс-баром
        all_tokens = []
        if self.progress_config.ENABLE_PROGRESS_BARS:
            for tokens in tqdm(train_df['tokens'], desc="Сбор токенов", 
                              bar_format=self.progress_config.BAR_FORMAT):
                all_tokens.extend(tokens)
        else:
            for tokens in train_df['tokens']:
                all_tokens.extend(tokens)
        
        word_counts = Counter(all_tokens)
        
        # Оставляем только частые слова
        print("Фильтрация редких слов...")
        vocab = {}
        idx = 0
        
        if self.progress_config.ENABLE_PROGRESS_BARS:
            for word, count in tqdm(word_counts.items(), desc="Создание словаря",
                                   bar_format=self.progress_config.BAR_FORMAT):
                if count >= self.config.MIN_WORD_COUNT:
                    vocab[word] = idx
                    idx += 1
        else:
            for word, count in word_counts.items():
                if count >= self.config.MIN_WORD_COUNT:
                    vocab[word] = idx
                    idx += 1
        
        # Добавляем токен для неизвестных слов
        if '<UNK>' not in vocab:
            vocab['<UNK>'] = len(vocab)
        
        vocab_size = len(vocab)
        print(f"\nРазмер словаря: {vocab_size}")
        print(f"Индекс <UNK>: {vocab.get('<UNK>', 'не найден')}")
        
        return vocab, vocab_size
    
    def tokens_to_indices(self, tokens, vocab):
        """Преобразование токенов в индексы с гарантией валидности."""
        unk_idx = vocab.get('<UNK>', 0)
        indices = []
        for token in tokens:
            idx = vocab.get(token, unk_idx)
            if idx >= len(vocab):
                idx = unk_idx
            indices.append(idx)
        return indices
    
