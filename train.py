import torch
import torch.nn as nn
from tqdm import tqdm  # Для красивого прогресс-бара
from config import DEVICE

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs, model_name):
    """
    ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ
    
    Args:
        model: Нейронная сеть для обучения
        train_loader: DataLoader для обучающих данных
        val_loader: DataLoader для валидационных данных
        optimizer: Оптимизатор (Adam, Lion и т.д.)
        criterion: Функция потерь
        num_epochs: Количество эпох обучения
        model_name: Имя модели для логирования
    
    Returns:
        train_losses, val_accuracies: История обучения для построения графиков
    """
    
    # Списки для хранения истории обучения (для графиков)
    train_losses = []
    val_accuracies = []
    
    print(f"\nНачало обучения: {model_name}")
    
    # ЦИКЛ ПО ЭПОХАМ
    for epoch in range(num_epochs):
        # ========== ФАЗА ОБУЧЕНИЯ ==========
        model.train()  # Переводим модель в режим обучения
        running_loss = 0.0  # Сумма потерь за эпоху
        
        # tqdm создает прогресс-бар для визуализации процесса
        for images, labels in tqdm(train_loader, desc=f'{model_name} - Эпоха {epoch+1}/{num_epochs}'):
            # Перенос данных на нужное устройство (GPU/CPU)
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # ОБНУЛЕНИЕ ГРАДИЕНТОВ - ВАЖНО делать перед каждым батчем!
            optimizer.zero_grad()
            
            # ПРЯМОЙ ПРОХОД (FORWARD PASS): предсказание модели
            outputs = model(images)
            
            # ВЫЧИСЛЕНИЕ ФУНКЦИИ ПОТЕРЬ (LOSS)
            loss = criterion(outputs, labels)
            
            # ОБРАТНЫЙ ПРОХОД (BACKWARD PASS): вычисление градиентов
            loss.backward()
            
            # ОБНОВЛЕНИЕ ВЕСОВ: шаг оптимизатора
            optimizer.step()
            
            # Суммируем потери для статистики
            running_loss += loss.item()
        
        # ========== ФАЗА ВАЛИДАЦИИ ==========
        model.eval()  # Переводим модель в режим оценки
        correct = 0   # Количество правильных предсказаний
        total = 0     # Общее количество примеров
        
        # torch.no_grad() отключает вычисление градиентов для экономии памяти
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                
                # torch.max возвращает максимальные значения и их индексы
                _, predicted = torch.max(outputs.data, 1)
                # _ - максимальные значения (нам не нужны)
                # predicted - предсказанные классы (индексы)
                
                total += labels.size(0)  # Увеличиваем счетчик примеров
                correct += (predicted == labels).sum().item()  # Считаем правильные
        
        # ВЫЧИСЛЕНИЕ МЕТРИК
        accuracy = 100 * correct / total  # Точность в процентах
        avg_loss = running_loss / len(train_loader)  # Средняя потеря за эпоху
        
        # Сохраняем метрики для графиков
        train_losses.append(avg_loss)
        val_accuracies.append(accuracy)
        
        # ВЫВОД СТАТИСТИКИ
        print(f'{model_name} - Эпоха [{epoch+1}/{num_epochs}], '
              f'Потери: {avg_loss:.4f}, Точность: {accuracy:.2f}%')
    
    print(f"Обучение завершено: {model_name}")
    return train_losses, val_accuracies