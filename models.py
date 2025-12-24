"""
Реализация моделей Word2Vec и нейронной сети на PyTorch с GPU поддержкой
"""
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from config import GPUConfig

class Word2VecPyTorch(nn.Module):
    """Реализация Word2Vec (Skip-gram) на PyTorch с GPU."""
    
    def __init__(self, vocab_size, embedding_dim, learning_rate=0.01, progress_config=None):
        super(Word2VecPyTorch, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.progress_config = progress_config
        
        # Эмбеддинги для входа и выхода
        self.in_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Инициализация весов
        self.in_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.out_embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        
        self.to(GPUConfig.DEVICE)
    
    def forward(self, target_idx, context_idx):
        """Прямой проход для пары слов."""
        target_embeds = self.in_embeddings(target_idx)  # [batch_size, embedding_dim]
        context_embeds = self.out_embeddings(context_idx)  # [batch_size, embedding_dim]
        
        # Вычисляем логиты (скалярное произведение)
        scores = torch.sum(target_embeds * context_embeds, dim=1)  # [batch_size]
        return scores
    
    def get_embeddings(self):
        """Возвращает матрицу эмбеддингов."""
        return self.in_embeddings.weight.detach().cpu().numpy()
    
    def train_model(self, training_pairs, epochs=5, batch_size=1000, verbose=True):
        """Обучение модели на GPU."""
        print(f"\nОбучение Word2Vec на {len(training_pairs)} парах...")
        
        if GPUConfig.USE_GPU:
            print(f"🎮 Используется GPU: {torch.cuda.get_device_name(0)}")
        
        # Преобразование данных в тензоры
        target_indices = torch.tensor([pair[0] for pair in training_pairs], dtype=torch.long)
        context_indices = torch.tensor([pair[1] for pair in training_pairs], dtype=torch.long)
        
        # Создание даталоадера
        dataset = TensorDataset(target_indices, context_indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Оптимизатор и функция потерь
        optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        
        # Обучение
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            batch_iterator = dataloader
            
            if verbose and self.progress_config and self.progress_config.ENABLE_PROGRESS_BARS:
                batch_iterator = tqdm(dataloader, 
                                    desc=f"Эпоха {epoch+1}/{epochs}",
                                    bar_format=self.progress_config.BAR_FORMAT)
            
            for target_batch, context_batch in batch_iterator:
                # Перемещение на GPU
                target_batch = target_batch.to(GPUConfig.DEVICE)
                context_batch = context_batch.to(GPUConfig.DEVICE)
                
                # Положительные примеры (target, context)
                pos_scores = self(target_batch, context_batch)
                pos_labels = torch.ones_like(pos_scores)
                
                # Негативные семплирование
                neg_indices = torch.randint(0, self.vocab_size, context_batch.shape, 
                                           dtype=torch.long, device=GPUConfig.DEVICE)
                neg_scores = self(target_batch, neg_indices)
                neg_labels = torch.zeros_like(neg_scores)
                
                # Объединяем положительные и отрицательные примеры
                all_scores = torch.cat([pos_scores, neg_scores])
                all_labels = torch.cat([pos_labels, neg_labels])
                
                # Вычисление потерь
                optimizer.zero_grad()
                loss = criterion(all_scores, all_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if verbose:
                avg_loss = total_loss / len(dataloader)
                print(f"Эпоха {epoch+1}/{epochs}, Потери: {avg_loss:.4f}")
        
        print("✓ Обучение Word2Vec завершено")
        return self.get_embeddings()


class NeuralNetworkPyTorch(nn.Module):
    """Улучшенная реализация нейронной сети на PyTorch с GPU."""
    
    def __init__(self, input_dim, hidden_dims, output_dim=1, progress_config=None):
        super().__init__()  # ИСПРАВЛЕНО: используется правильный синтаксис
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.progress_config = progress_config
        
        # Создание слоев с нормализацией и регуляризацией
        layers = []
        current_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Полносвязный слой
            layers.append(nn.Linear(current_dim, hidden_dim))
            
            # Пакетная нормализация (только если batch_size > 1)
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            # Функция активации
            layers.append(nn.ReLU())
            
            # Dropout регуляризация
            dropout_rate = 0.5 if i == 0 else 0.3  # Больше dropout в первом слое
            layers.append(nn.Dropout(dropout_rate))
            
            current_dim = hidden_dim
        
        # Выходной слой
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        self.to(GPUConfig.DEVICE)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Инициализация весов для лучшей сходимости."""
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.model(x)
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, 
                   epochs=100, learning_rate=0.001, batch_size=64, verbose=True):
        """Обучение нейронной сети на GPU с ранней остановкой."""
        if GPUConfig.USE_GPU:
            print(f"🎮 Используется GPU для обучения нейронной сети")
        
        # Преобразование данных в тензоры PyTorch
        X_train_tensor = torch.FloatTensor(X_train).to(GPUConfig.DEVICE)
        y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(GPUConfig.DEVICE)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(GPUConfig.DEVICE)
            y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(GPUConfig.DEVICE)
        
        # Создание даталоадеров
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        # Настройки для ранней остановки
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
        
        # Обучение
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0
            batch_iterator = train_loader
            
            if verbose and self.progress_config and self.progress_config.ENABLE_PROGRESS_BARS:
                batch_iterator = tqdm(train_loader,
                                     desc=f"Эпоха {epoch+1}/{epochs}",
                                     bar_format=self.progress_config.BAR_FORMAT,
                                     leave=False)
            
            for X_batch, y_batch in batch_iterator:
                optimizer.zero_grad()
                y_pred = self(X_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * X_batch.size(0)
            
            # Средние потери за эпоху
            epoch_loss /= len(X_train)
            train_losses.append(epoch_loss)
            
            # Валидация
            if X_val is not None and y_val is not None:
                self.eval()
                with torch.no_grad():
                    y_val_pred = self(X_val_tensor)
                    val_loss = criterion(y_val_pred, y_val_tensor).item()
                    val_losses.append(val_loss)
                    
                    val_pred_labels = (y_val_pred > 0.5).float()
                    val_accuracy = (val_pred_labels == y_val_tensor).float().mean().item()
                    val_accuracies.append(val_accuracy)
                
                # Ранняя остановка
                if val_loss < best_val_loss - 0.001:  # Минимальное улучшение 0.001
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Сохраняем лучшие веса модели
                    best_model_state = self.state_dict().copy()
                else:
                    patience_counter += 1
                
                self.train()
            
            # Вывод информации
            if verbose and not self.progress_config.ENABLE_PROGRESS_BARS and (epoch + 1) % 10 == 0:
                if X_val is not None:
                    print(f"Эпоха {epoch+1}/{epochs}, Потери: {epoch_loss:.4f}, "
                          f"Валидационные потери: {val_loss:.4f}, Точность: {val_accuracy:.4f}")
                else:
                    print(f"Эпоха {epoch+1}/{epochs}, Потери: {epoch_loss:.4f}")
            
            # Проверка ранней остановки
            if patience_counter >= patience:
                print(f"\n⚠️ Ранняя остановка на эпохе {epoch+1} (без улучшения {patience} эпох)")
                # Загружаем лучшие веса модели
                if best_model_state is not None:
                    self.load_state_dict(best_model_state)
                break
        
        return train_losses, val_losses, val_accuracies
    
    def predict(self, X):
        """Предсказание на новых данных."""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(GPUConfig.DEVICE)
            y_pred = self(X_tensor)
            y_pred_labels = (y_pred > 0.5).float()
            
        return y_pred_labels.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten()