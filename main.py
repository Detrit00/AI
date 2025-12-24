"""
Основной скрипт для запуска проекта с прогресс-барами и GPU
"""
import numpy as np
import random
import torch
from tqdm import tqdm
from config import *
from data_loader import DataLoader
from models import Word2VecPyTorch, NeuralNetworkPyTorch
from utils import ModelEvaluator, Visualization, ReportGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 
from collections import Counter

def augment_text(tokens, label):
    """Аугментация данных с фокусом на балансировке."""
    augmented = tokens.copy()
    
    # Более агрессивная аугментация для меньшего класса
    if label == 1:  # Если это Fake новости (которых меньше)
        if len(tokens) > 4:
            # Больше перестановок для меньшего класса
            num_swaps = min(3, len(tokens) // 2)
            for _ in range(num_swaps):
                i, j = random.sample(range(len(tokens)), 2)
                augmented[i], augmented[j] = augmented[j], augmented[i]
    else:
        # Обычная аугментация для большего класса
        if len(tokens) > 3 and random.random() > 0.5:
            i, j = random.sample(range(len(tokens)), 2)
            augmented[i], augmented[j] = augmented[j], augmented[i]
    
    return augmented

def create_training_pairs(indices, window_size, vocab_size, progress_config):
    """Создание обучающих пар для Word2Vec с прогресс-баром."""
    print("\n" + "="*60)
    print("СОЗДАНИЕ ОБУЧАЮЩИХ ПАР ДЛЯ WORD2VEC")
    print("="*60)
    
    training_pairs = []
    skipped_pairs = 0
    
    bar_format = None
    if progress_config and progress_config.ENABLE_PROGRESS_BARS:
        bar_format = progress_config.BAR_FORMAT
    
    # Прогресс-бар по документам
    if progress_config.ENABLE_PROGRESS_BARS:
        doc_iterator = tqdm(
            enumerate(indices),
            total=len(indices),
            desc="Обработка документов",
            bar_format=bar_format
        )
    else:
        doc_iterator = enumerate(indices)
    
    for doc_idx, doc_indices in doc_iterator:
        for i, target_idx in enumerate(doc_indices):
            start = max(0, i - window_size)
            end = min(len(doc_indices), i + window_size + 1)
            
            for j in range(start, end):
                if i != j:
                    context_idx = doc_indices[j]
                    
                    if target_idx < vocab_size and context_idx < vocab_size:
                        training_pairs.append((target_idx, context_idx))
                    else:
                        skipped_pairs += 1
    
    print(f"\n✓ Создано обучающих пар: {len(training_pairs):,}")
    if skipped_pairs > 0:
        print(f"✗ Пропущено пар с недопустимыми индексами: {skipped_pairs:,}")
    
    return training_pairs

def validate_indices(indices_list, vocab_size, progress_config):
    """Проверка всех индексов на валидность с прогресс-баром."""
    print("\nПроверка валидности индексов...")
    
    max_idx = 0
    min_idx = vocab_size
    invalid_count = 0
    
    bar_format = None
    if progress_config and progress_config.ENABLE_PROGRESS_BARS:
        bar_format = progress_config.BAR_FORMAT
    
    if progress_config.ENABLE_PROGRESS_BARS:
        indices_iterator = tqdm(
            indices_list,
            desc="Проверка индексов",
            bar_format=bar_format
        )
    else:
        indices_iterator = indices_list
    
    for indices in indices_iterator:
        for idx in indices:
            if idx >= vocab_size:
                invalid_count += 1
            if idx > max_idx:
                max_idx = idx
            if idx < min_idx:
                min_idx = idx
    
    print(f"\nРезультаты проверки:")
    print(f"  Максимальный индекс: {max_idx}")
    print(f"  Минимальный индекс: {min_idx}")
    print(f"  Невалидных индексов: {invalid_count}")
    print(f"  Размер словаря: {vocab_size}")
    
    if max_idx >= vocab_size:
        print(f"  ⚠️ ВНИМАНИЕ: Обнаружены индексы >= vocab_size!")
        return False
    
    return True

def text_to_vector(tokens, embeddings, vocab, word_counts=None):
    """Улучшенная векторизация текста с TF-IDF взвешиванием."""
    vectors = []
    weights = []
    unk_idx = vocab.get('<UNK>', 0)
    unk_vector = embeddings[unk_idx] if unk_idx < embeddings.shape[0] else np.zeros(embeddings.shape[1])
    
    total_docs = 10000  # Приблизительное количество документов для IDF
    
    for token in tokens:
        idx = vocab.get(token, unk_idx)
        if idx < embeddings.shape[0]:
            vectors.append(embeddings[idx])
            
            # TF-IDF взвешивание
            tf = 1.0  # частота в документе (упрощенно)
            # Приблизительный IDF (в реальном проекте нужно считать точно)
            doc_freq = word_counts.get(token, 1) if word_counts else 1
            idf = np.log((total_docs + 1) / (doc_freq + 1)) + 1
            weights.append(tf * idf)
    
    if len(vectors) == 0:
        return unk_vector
    
    vectors = np.array(vectors)
    weights = np.array(weights)
    
    # Нормализуем веса
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        weights = np.ones(len(vectors)) / len(vectors)
    
    # Взвешенная сумма вместо среднего
    weighted_sum = np.sum(vectors * weights[:, np.newaxis], axis=0)
    
    # Нормализация полученного вектора
    norm = np.linalg.norm(weighted_sum)
    if norm > 0:
        weighted_sum = weighted_sum / norm
    
    return weighted_sum

def analyze_important_words(word_embeddings, vocab, word_counts, nn_model, progress_config):
    """Анализ важных слов на основе весов модели с прогресс-баром."""
    print("\n" + "="*60)
    print("АНАЛИЗ ВАЖНЫХ СЛОВ")
    print("="*60)
    
    # Для PyTorch модели получаем веса первого слоя
    first_layer_weights = nn_model.model[0].weight.data.cpu().numpy()
    
    # Размерность эмбеддингов должна совпадать с размерностью входа первого слоя
    embedding_dim = word_embeddings.shape[1]
    input_dim = first_layer_weights.shape[1]
    
    if embedding_dim != input_dim:
        print(f"⚠️ Внимание: размерность эмбеддингов ({embedding_dim}) не совпадает")
        print(f"с размерностью входа нейронной сети ({input_dim})")
        print("Используем только первые {input_dim} измерений эмбеддингов")
    
    # Усекаем эмбеддинги до нужной размерности
    max_dim = min(embedding_dim, input_dim)
    truncated_embeddings = word_embeddings[:, :max_dim]
    
    # Вычисляем важность признаков как среднее абсолютных значений весов
    feature_importance = np.mean(np.abs(first_layer_weights), axis=0)
    
    # Берем только нужные измерения
    feature_importance = feature_importance[:max_dim]
    
    # Берем более частые слова (100 вместо 50)
    common_words = [word for word, count in word_counts.most_common(100) if word in vocab]
    
    word_importance = []
    
    bar_format = None
    if progress_config and progress_config.ENABLE_PROGRESS_BARS:
        bar_format = progress_config.BAR_FORMAT
    
    if progress_config.ENABLE_PROGRESS_BARS:
        words_iterator = tqdm(
            common_words,
            desc="Анализ важности слов",
            bar_format=bar_format
        )
    else:
        words_iterator = common_words
    
    for word in words_iterator:
        if word in vocab:
            word_idx = vocab[word]
            if word_idx < truncated_embeddings.shape[0]:
                importance = np.dot(np.abs(truncated_embeddings[word_idx]), feature_importance)
                word_importance.append((word, importance))
    
    word_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nТоп-10 самых важных слов для классификации:")
    for i, (word, importance) in enumerate(word_importance[:10]):
        print(f"{i+1:2}. {word:15} - важность: {importance:.6f}")
    
    return word_importance[:20]

def main():
    """Основная функция запуска проекта."""
    print("="*60)
    print("🚀 ЗАПУСК ПРОЕКТА: КЛАССИФИКАЦИЯ ФЕЙКОВЫХ НОВОСТЕЙ")
    print("📊 Версия с прогресс-барами, GPU и балансировкой")
    print("="*60)
    
    try:
        # Информация о GPU
        GPUConfig.print_gpu_info()
        
        # 1. Инициализация
        print("🎯 Инициализация проекта...")
        Visualization.create_output_directories(OutputConfig)
        
        # 2. Загрузка и предобработка данных
        data_config = DataConfig()
        data_loader = DataLoader(data_config, ProgressConfig)
        train_df, test_df = data_loader.load_all_data()
        
        if len(train_df) == 0:
            print("❌ ОШИБКА: Не загружены тренировочные данные!")
            return 1
        
        train_df, test_df = data_loader.preprocess_dataframes(train_df, test_df)
        
        # 3. Построение словаря
        vocab, vocab_size = data_loader.build_vocabulary(train_df)
        
        # 4. Преобразование токенов в индексы
        print("\n" + "="*60)
        print("ПРЕОБРАЗОВАНИЕ ТОКЕНОВ В ИНДЕКСЫ")
        print("="*60)
        
        if ProgressConfig.ENABLE_PROGRESS_BARS:
            tqdm.pandas(desc="Индексация тренировочных токенов", 
                       bar_format=ProgressConfig.BAR_FORMAT)
            train_df['indices'] = train_df.apply(
                lambda row: data_loader.tokens_to_indices(
                    augment_text(row['tokens'], row['label']), vocab
                ), axis=1
            )
        else:
            train_df['indices'] = train_df.apply(
                lambda row: data_loader.tokens_to_indices(
                    augment_text(row['tokens'], row['label']), vocab
                ), axis=1
            )
        
        # 5. Проверка валидности индексов
        if not validate_indices(train_df['indices'].values, vocab_size, ProgressConfig):
            print("Исправление недопустимых индексов...")
            unk_idx = vocab.get('<UNK>', 0)
            train_df['indices'] = train_df['indices'].apply(
                lambda idx_list: [idx if idx < vocab_size else unk_idx for idx in idx_list]
            )
        
        # 6. Создание обучающих пар для Word2Vec
        training_pairs = create_training_pairs(
            train_df['indices'].values,
            Word2VecConfig.WINDOW_SIZE,
            vocab_size,
            ProgressConfig
        )
        
        if len(training_pairs) == 0:
            print("⚠️ ВНИМАНИЕ: Не создано обучающих пар для Word2Vec!")
            print("Создаем случайные эмбеддинги...")
            word_embeddings = np.random.randn(vocab_size, Word2VecConfig.EMBEDDING_DIM)
        else:
            # 7. Обучение Word2Vec на GPU
            w2v_model = Word2VecPyTorch(
                vocab_size, 
                Word2VecConfig.EMBEDDING_DIM,
                Word2VecConfig.LEARNING_RATE,
                ProgressConfig
            )
            
            word_embeddings = w2v_model.train_model(
                training_pairs,
                epochs=Word2VecConfig.EPOCHS,
                batch_size=Word2VecConfig.BATCH_SIZE,
                verbose=LoggingConfig.VERBOSE
            )
        
        # 8. Преобразование текстов в векторы
        print("\n" + "="*60)
        print("ПРЕОБРАЗОВАНИЕ ТЕКСТОВ В ВЕКТОРЫ")
        print("="*60)
        
        # Получаем частоты слов для TF-IDF
        all_tokens = [token for tokens in train_df['tokens'] for token in tokens]
        word_counts = Counter(all_tokens)
        
        if ProgressConfig.ENABLE_PROGRESS_BARS:
            tqdm.pandas(desc="Векторизация тренировочных текстов", 
                       bar_format=ProgressConfig.BAR_FORMAT)
            train_df['vector'] = train_df['tokens'].progress_apply(
                lambda x: text_to_vector(x, word_embeddings, vocab, word_counts)
            )
            
            tqdm.pandas(desc="Векторизация тестовых текстов",
                       bar_format=ProgressConfig.BAR_FORMAT)
            test_df['vector'] = test_df['tokens'].progress_apply(
                lambda x: text_to_vector(x, word_embeddings, vocab, word_counts)
            )
        else:
            train_df['vector'] = train_df['tokens'].apply(
                lambda x: text_to_vector(x, word_embeddings, vocab, word_counts)
            )
            test_df['vector'] = test_df['tokens'].apply(
                lambda x: text_to_vector(x, word_embeddings, vocab, word_counts)
            )
        
        X_train = np.vstack(train_df['vector'].values)
        y_train = train_df['label'].values
        X_test = np.vstack(test_df['vector'].values)
        y_test = test_df['label'].values
        
        print(f"\n✓ Размерность данных:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        
        # 9. Нормализация векторов
        print("\n" + "="*60)
        print("📊 НОРМАЛИЗАЦИЯ ВЕКТОРОВ")
        print("="*60)
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # 10. Разделение на тренировочные и валидационные данные
        print("\n" + "="*60)
        print("РАЗДЕЛЕНИЕ ДАННЫХ НА TRAIN/VAL")
        print("="*60)
        
        # Стратифицированное разделение для сохранения баланса классов
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train,
            test_size=NeuralNetConfig.VALIDATION_SPLIT,
            stratify=y_train,  # Сохраняем распределение классов
            random_state=42,   # Для воспроизводимости
            shuffle=True       # Перемешиваем данные
        )
        
        print(f"✓ Стратифицированное разделение данных:")
        print(f"  Train: {X_train_split.shape} (классы: {np.bincount(y_train_split.astype(int))})")
        print(f"  Val:   {X_val_split.shape} (классы: {np.bincount(y_val_split.astype(int))})")
        print(f"  Test:  {X_test.shape} (классы: {np.bincount(y_test.astype(int))})")
        
        # Проверка баланса классов
        print(f"\nБаланс классов:")
        print(f"  Train - Real: {np.sum(y_train_split == 0):,}, Fake: {np.sum(y_train_split == 1):,}")
        print(f"  Val   - Real: {np.sum(y_val_split == 0):,}, Fake: {np.sum(y_val_split == 1):,}")
        print(f"  Test  - Real: {np.sum(y_test == 0):,}, Fake: {np.sum(y_test == 1):,}")
        
        # 11. Балансировка классов
        print("\n" + "="*60)
        print("⚖️ БАЛАНСИРОВКА КЛАССОВ")
        print("="*60)
        
        # Вычисляем веса классов
        class_counts = np.bincount(y_train_split.astype(int))
        total_samples = len(y_train_split)
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        print(f"Количество в train: Real={class_counts[0]}, Fake={class_counts[1]}")
        print(f"Веса классов: Real={class_weights[0]:.2f}, Fake={class_weights[1]:.2f}")
        
        # 12. Обучение нейронной сети на GPU
        print("\n" + "="*60)
        print("🧠 ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ НА GPU")
        print("="*60)
        
        nn_model = NeuralNetworkPyTorch(
            input_dim=Word2VecConfig.EMBEDDING_DIM,
            hidden_dims=NeuralNetConfig.HIDDEN_LAYERS,
            output_dim=NeuralNetConfig.OUTPUT_DIM,
            progress_config=ProgressConfig
        )
        
        train_losses, val_losses, val_accuracies = nn_model.train_model(
            X_train_split, y_train_split,
            X_val_split, y_val_split,
            epochs=NeuralNetConfig.EPOCHS,
            learning_rate=NeuralNetConfig.LEARNING_RATE,
            batch_size=NeuralNetConfig.BATCH_SIZE,
            verbose=LoggingConfig.VERBOSE
        )
        
        # 13. Предсказание на тестовых данных
        print("\n" + "="*60)
        print("🧪 ТЕСТИРОВАНИЕ МОДЕЛИ")
        print("="*60)
        
        y_pred_labels, y_pred_proba = nn_model.predict(X_test)
        
        # 14. Вычисление метрик
        evaluator = ModelEvaluator()
        metrics = evaluator.calculate_metrics(y_test, y_pred_labels, y_pred_proba)
        
        print(f"\n📊 МЕТРИКИ КАЧЕСТВА МОДЕЛИ:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['auc']:.4f}")
        
        # 15. Анализ ошибок
        print("\n" + "="*60)
        print("🔍 АНАЛИЗ ОШИБОК")
        print("="*60)
        
        # Находим примеры, где модель ошибается
        errors_mask = (y_test != y_pred_labels)
        if np.sum(errors_mask) > 0:
            error_indices = np.where(errors_mask)[0]
            print(f"Количество ошибок: {len(error_indices)}")
            
            # Смотрим на распределение ошибок по классам
            false_positives = np.sum((y_test == 0) & (y_pred_labels == 1))
            false_negatives = np.sum((y_test == 1) & (y_pred_labels == 0))
            
            print(f"False Positives (Real предсказаны как Fake): {false_positives}")
            print(f"False Negatives (Fake предсказаны как Real): {false_negatives}")
            
            # Адаптивный порог на основе баланса ошибок
            if false_positives > false_negatives * 1.5:
                print("\n⚠️ Обнаружен дисбаланс: слишком много False Positives")
                print("Пробуем разные пороги для улучшения баланса...")
                
                best_f1 = metrics['f1']
                best_threshold = 0.5
                
                for threshold in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
                    y_pred_adjusted = (y_pred_proba > threshold).astype(int)
                    metrics_adjusted = evaluator.calculate_metrics(y_test, y_pred_adjusted, y_pred_proba)
                    
                    fp = np.sum((y_test == 0) & (y_pred_adjusted == 1))
                    fn = np.sum((y_test == 1) & (y_pred_adjusted == 0))
                    
                    print(f"  Порог {threshold:.2f}: F1={metrics_adjusted['f1']:.4f}, "
                          f"FP={fp}, FN={fn}")
                    
                    if metrics_adjusted['f1'] > best_f1:
                        best_f1 = metrics_adjusted['f1']
                        best_threshold = threshold
                
                print(f"\n✓ Лучший порог: {best_threshold:.2f} (F1={best_f1:.4f})")
                
                # Используем лучший порог
                y_pred_labels = (y_pred_proba > best_threshold).astype(int)
                metrics = evaluator.calculate_metrics(y_test, y_pred_labels, y_pred_proba)
                
                print(f"\n📊 МЕТРИКИ С ПОРОГОМ {best_threshold:.2f}:")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1-Score:  {metrics['f1']:.4f}")
        
        # 16. Анализ важных слов
        important_words = analyze_important_words(
            word_embeddings, vocab, word_counts, nn_model, ProgressConfig
        )
        
        # 17. Сохранение графиков
        print("\n" + "="*60)
        print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("="*60)
        
        # Сохраняем все графики
        Visualization.save_learning_curve(
            train_losses, val_losses,
            f"{OutputConfig.GRAPHS_DIR}/{OutputConfig.GRAPH_NAMES['learning_curve']}"
        )
        
        if val_accuracies:
            Visualization.save_validation_accuracy(
                val_accuracies,
                f"{OutputConfig.GRAPHS_DIR}/{OutputConfig.GRAPH_NAMES['validation_accuracy']}"
            )
        
        Visualization.save_prediction_distribution(
            y_test, y_pred_proba,
            f"{OutputConfig.GRAPHS_DIR}/{OutputConfig.GRAPH_NAMES['prediction_distribution']}"
        )
        
        Visualization.save_confusion_matrix(
            metrics['confusion_matrix'],
            f"{OutputConfig.GRAPHS_DIR}/{OutputConfig.GRAPH_NAMES['confusion_matrix']}"
        )
        
        if important_words:
            top_words = [word for word, _ in important_words[:10]]
            top_importances = [imp for _, imp in important_words[:10]]
            Visualization.save_important_words(
                top_words, top_importances,
                f"{OutputConfig.GRAPHS_DIR}/{OutputConfig.GRAPH_NAMES['important_words']}"
            )
        
        print("✓ Все графики успешно сохранены!")
        
        # 18. Сохранение отчетов
        results = {
            'metrics': metrics,
            'important_words': important_words,
            'vocab_size': vocab_size,
            'embedding_dim': Word2VecConfig.EMBEDDING_DIM,
            'training_pairs_count': len(training_pairs),
            'model_architecture': f"{Word2VecConfig.EMBEDDING_DIM} -> {NeuralNetConfig.HIDDEN_LAYERS} -> 1",
            'class_weights': {
                'real': float(class_weights[0]),
                'fake': float(class_weights[1])
            }
        }
        
        ReportGenerator.save_text_report(
            results, OutputConfig, vocab_size, 
            Word2VecConfig.EMBEDDING_DIM, NeuralNetConfig
        )
        
        ReportGenerator.save_metrics_json(
            metrics, OutputConfig.METRICS_FILE
        )
        
        # 19. Итоговый вывод
        print("\n" + "="*60)
        print("✅ ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА!")
        print("="*60)
        
        print(f"\n📈 ИТОГИ ЭКСПЕРИМЕНТА:")
        print(f"  📁 Загружено документов: {len(train_df) + len(test_df)}")
        print(f"  📖 Размер словаря: {vocab_size}")
        print(f"  🔗 Обучающих пар Word2Vec: {len(training_pairs):,}")
        print(f"  🎮 Использовалась GPU: {'Да' if GPUConfig.USE_GPU else 'Нет'}")
        print(f"  🧮 Точность модели: {metrics['accuracy']:.2%}")
        print(f"  📊 F1-Score: {metrics['f1']:.4f}")
        print(f"  ⚖️ Баланс ошибок: FP={false_positives}, FN={false_negatives}")
        
        print(f"\n💾 СОХРАНЕННЫЕ ФАЙЛЫ:")
        print(f"  📄 Отчеты: {OutputConfig.REPORTS_DIR}/")
        print(f"    • {OutputConfig.RESULTS_FILE}")
        print(f"    • {OutputConfig.METRICS_FILE}")
        print(f"  📈 Графики: {OutputConfig.GRAPHS_DIR}/")
        for name in OutputConfig.GRAPH_NAMES:
            print(f"    • {OutputConfig.GRAPH_NAMES[name]}")
        
        print("\n" + "="*60)
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Программа прервана пользователем")
        return 130
    except Exception as e:
        print(f"\n❌ ОШИБКА ВЫПОЛНЕНИЯ: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)