from sklearn.metrics import precision_score, recall_score, classification_report, confusion_matrix
import torch
import matplotlib.pyplot as plt
import numpy as np
from config import DEVICE

def evaluate_model(model, test_loader, model_name):
    """оценка модели на тест данных
    Вычисляет Precision, Recall и другие метрики
    
    Args:
        model: Обученная модель
        test_loader: DataLoader тестовых данных
        model_name: Имя модели для вывода
    
    Returns:
        precision, recall: Значения метрик
    """
    
    model.eval()  #оценки
    all_predictions = []  #предсказания модели
    all_labels = [] #настоящие метки
    
    print(f"\nОценка модели: {model_name}")
    
    #Количество уникальных классов в тестовых данных
    unique_labels = set()
    for _, labels in test_loader:
        unique_labels.update(labels.numpy())
    
    print(f"В тестовом наборе представлено {len(unique_labels)} уникальных классов")
    
    #сбор предсказаний для всех тестовых данных
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            #сохраняем предсказания и метки(переводим в numpy для sklearn)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    #проверка что у нас есть все классы
    unique_test_labels = set(all_labels)
    unique_test_preds = set(all_predictions)
    print(f"Фактически использовано меток: {len(unique_test_labels)} классов")
    print(f"Фактически предсказано: {len(unique_test_preds)} классов")
    
    #вычисление с помощью sklearn
    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    #'weighted' означает взвешенное среднее по классам
    #zero_division=0 обрабатывает случаи, когда нет предсказаний для класса
    
    #Результаты
    print(f"\n=== РЕЗУЛЬТАТЫ {model_name} ===")
    print(f"Precision: {precision:.4f}")  #точность: сколько из предсказанных правильные
    print(f"Recall: {recall:.4f}")        #полнота: сколько из настоящих нашли
    print("\nДетальный отчет по классам:")
    print(classification_report(all_labels, all_predictions, zero_division=0))
    
    return precision, recall

def plot_results(results_dict):
    """
    результаты экспериментов 
    
    три графика:
    1.График потерь при обучении
    2.График точности на валидации
    3.Сравнение Precision и Recall
    
    
        results_dict: Словарь с результатами всех моделей
    """
    
    #фигура с тремя subplot
    plt.figure(figsize=(15, 5))
    
    #график 1 поетри при обучении
    plt.subplot(1, 3, 1)
    for model_name, results in results_dict.items():
        plt.plot(results['train_losses'], label=model_name, linewidth=2)
    
    plt.title('функция потерь при обучении', fontsize=14, fontweight='bold')
    plt.xlabel('эпоха')
    plt.ylabel('потери')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # график 2: точность валидации
    plt.subplot(1, 3, 2)
    for model_name, results in results_dict.items():
        plt.plot(results['val_accuracies'], label=model_name, linewidth=2)
    
    plt.title('Точность на валидации', fontsize=14, fontweight='bold')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # график 3 сравнение PRECISION И RECALL
    plt.subplot(1, 3, 3)
    model_names = list(results_dict.keys())
    precisions = [results_dict[name]['precision'] for name in model_names]
    recalls = [results_dict[name]['recall'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35  #Ширина столбцов
    
    #столбчатая диаграмма для сравнения метрик
    plt.bar(x - width/2, precisions, width, label='Precision', alpha=0.7, color='skyblue')
    plt.bar(x + width/2, recalls, width, label='Recall', alpha=0.7, color='lightcoral')
    
    plt.title('Сравнение Precision и Recall', fontsize=14, fontweight='bold')
    plt.xlabel('Модели')
    plt.ylabel('Значение метрики')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    #настройка внешнего вида
    plt.tight_layout()
    
    #сохранение графика в файл
    plt.savefig('./results/comparison.png', dpi=300, bbox_inches='tight')
    print("\nГрафики сохранены в файл: ./results/comparison.png")
    
    #показ графика
    plt.show()

def print_final_comparison(results_dict):
    """
    Итоговая таблица
    """
    print("\n" + "="*70)
    print("ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    print(f"{'Модель':<25} {'Precision':<10} {'Recall':<10} {'Max Val Accuracy':<15}")
    print("-"*70)
    
    for model_name, results in results_dict.items():
        max_accuracy = max(results['val_accuracies'])
        print(f"{model_name:<25} {results['precision']:<10.4f} {results['recall']:<10.4f} {max_accuracy:<15.2f}%")
    
    print("="*70)