"""
Вспомогательные функции для оценки, визуализации и сохранения результатов
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class ModelEvaluator:
    """Класс для оценки модели и вычисления метрик."""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_pred_proba):
        """Вычисление всех метрик классификации."""
        # Матрица ошибок
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Основные метрики
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # ROC-AUC
        auc = ModelEvaluator.calculate_roc_auc(y_true, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': np.array([[tn, fp], [fn, tp]]),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    
    @staticmethod
    def calculate_roc_auc(y_true, y_pred_proba):
        """Вычисление ROC-AUC."""
        sorted_indices = np.argsort(-y_pred_proba)
        y_true_sorted = y_true[sorted_indices]
        
        tpr_values = []
        fpr_values = []
        
        for i in range(len(y_true_sorted) + 1):
            pred_labels = np.zeros_like(y_true_sorted)
            if i > 0:
                pred_labels[:i] = 1
            
            tp_i = np.sum((y_true_sorted == 1) & (pred_labels == 1))
            fp_i = np.sum((y_true_sorted == 0) & (pred_labels == 1))
            tn_i = np.sum((y_true_sorted == 0) & (pred_labels == 0))
            fn_i = np.sum((y_true_sorted == 1) & (pred_labels == 0))
            
            tpr = tp_i / (tp_i + fn_i) if (tp_i + fn_i) > 0 else 0
            fpr = fp_i / (fp_i + tn_i) if (fp_i + tn_i) > 0 else 0
            
            tpr_values.append(tpr)
            fpr_values.append(fpr)
        
        auc = 0
        for i in range(1, len(fpr_values)):
            auc += (fpr_values[i] - fpr_values[i-1]) * (tpr_values[i] + tpr_values[i-1]) / 2
        
        return auc


class Visualization:
    """Класс для создания и сохранения графиков."""
    
    @staticmethod
    def create_output_directories(output_config):
        """Создание папок для результатов."""
        os.makedirs(output_config.GRAPHS_DIR, exist_ok=True)
        os.makedirs(output_config.REPORTS_DIR, exist_ok=True)
        print(f"✓ Созданы папки для результатов")
        print(f"  • {output_config.GRAPHS_DIR}")
        print(f"  • {output_config.REPORTS_DIR}")
    
    @staticmethod
    def save_learning_curve(train_losses, val_losses, filepath):
        """Сохранение графика кривой обучения."""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss', linewidth=2)
        if val_losses:
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Learning Curve', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def save_validation_accuracy(val_accuracies, filepath):
        """Сохранение графика точности на валидации."""
        if not val_accuracies:
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(val_accuracies, linewidth=2, color='green')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Validation Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def save_prediction_distribution(y_test, y_pred_proba, filepath):
        """Сохранение распределения предсказанных вероятностей."""
        real_probs = y_pred_proba[y_test == 0]
        fake_probs = y_pred_proba[y_test == 1]
        
        plt.figure(figsize=(10, 6))
        plt.hist(real_probs, bins=20, alpha=0.7, label='Real News', 
                color='blue', density=True, edgecolor='black')
        plt.hist(fake_probs, bins=20, alpha=0.7, label='Fake News', 
                color='red', density=True, edgecolor='black')
        
        plt.xlabel('Predicted Probability (Fake)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Predictions', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def save_confusion_matrix(cm, filepath):
        """Сохранение тепловой карты матрицы ошибок."""
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix', fontsize=14)
        plt.colorbar()
        
        classes = ['Real', 'Fake']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, fontsize=12)
        plt.yticks(tick_marks, classes, fontsize=12)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=14)
        
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def save_important_words(words, importances, filepath):
        """Сохранение графика важных слов."""
        plt.figure(figsize=(12, 6))
        plt.barh(range(len(words)), importances[::-1], color='steelblue', edgecolor='black')
        plt.yticks(range(len(words)), words[::-1], fontsize=11)
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Important Words for Classification', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()


class ReportGenerator:
    """Класс для генерации отчетов."""
    
    @staticmethod
    def save_text_report(results, output_config, vocab_size, embedding_dim, nn_config):
        """Сохранение текстового отчета."""
        with open(output_config.RESULTS_FILE, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА: КЛАССИФИКАЦИЯ ФЕЙКОВЫХ НОВОСТЕЙ\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. ОБЩАЯ ИНФОРМАЦИЯ\n")
            f.write("-"*40 + "\n")
            f.write(f"Размер словаря: {vocab_size}\n")
            f.write(f"Размерность эмбеддингов: {embedding_dim}\n")
            f.write(f"Архитектура сети: {embedding_dim} -> {nn_config.HIDDEN_LAYERS} -> 1\n")
            f.write(f"Эпох обучения Word2Vec: {results.get('w2v_epochs', 'N/A')}\n")
            f.write(f"Эпох обучения нейронной сети: {nn_config.EPOCHS}\n\n")
            
            f.write("2. МЕТРИКИ КАЧЕСТВА\n")
            f.write("-"*40 + "\n")
            metrics = results.get('metrics', {})
            f.write(f"Accuracy:  {metrics.get('accuracy', 0):.4f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall:    {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score:  {metrics.get('f1', 0):.4f}\n")
            f.write(f"ROC-AUC:   {metrics.get('auc', 0):.4f}\n\n")
            
            f.write("3. МАТРИЦА ОШИБОК\n")
            f.write("-"*40 + "\n")
            cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
            f.write(f"              Predicted\n")
            f.write(f"              Real   Fake\n")
            f.write(f"Actual Real   {cm[0][0]:4d}   {cm[0][1]:4d}\n")
            f.write(f"        Fake   {cm[1][0]:4d}   {cm[1][1]:4d}\n\n")
            
            f.write("4. ВАЖНЫЕ СЛОВА\n")
            f.write("-"*40 + "\n")
            important_words = results.get('important_words', [])
            for i, (word, importance) in enumerate(important_words[:20]):
                f.write(f"{i+1:2d}. {word:20} - важность: {importance:.6f}\n")
            
            f.write("\n5. СОЗДАННЫЕ ФАЙЛЫ\n")
            f.write("-"*40 + "\n")
            for name, filename in output_config.GRAPH_NAMES.items():
                f.write(f"{name}: {filename}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("ЭКСПЕРИМЕНТ ЗАВЕРШЕН\n")
            f.write("="*80 + "\n")
        
        print(f"Текстовый отчет сохранен: {output_config.RESULTS_FILE}")
    
    @staticmethod
    def save_metrics_json(metrics, filepath):
        """Сохранение метрик в JSON формате."""
        metrics_dict = {
            'accuracy': float(metrics.get('accuracy', 0)),
            'precision': float(metrics.get('precision', 0)),
            'recall': float(metrics.get('recall', 0)),
            'f1': float(metrics.get('f1', 0)),
            'auc': float(metrics.get('auc', 0)),
            'confusion_matrix': metrics.get('confusion_matrix', [[0, 0], [0, 0]]).tolist()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"Метрики в JSON сохранены: {filepath}")