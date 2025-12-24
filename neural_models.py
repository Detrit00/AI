"""
Полноценные нейросетевые модели для оценки и генерации стихов
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
import os

# Проверяем доступность transformers
try:
    from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers не установлен. Используется упрощенная версия.")
    TRANSFORMERS_AVAILABLE = False

class PoetryCritic(nn.Module):
    """Нейросеть-критик для оценки качества стихов"""
    
    def __init__(self, vocab_size=20000, embedding_dim=128, hidden_dim=256):
        super(PoetryCritic, self).__init__()
        
        # Эмбеддинги для русского языка
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM для анализа контекста
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=0.3
        )
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(
            hidden_dim * 2,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )
        
        # Heads для разных аспектов оценки
        self.rhyme_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.meter_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.style_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
            nn.Softmax(dim=1)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, x, attention_mask=None):
        # Эмбеддинги
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, _ = self.lstm(embedded)
        
        # Attention
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attended, _ = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=attention_mask
        )
        
        # Pooling (усреднение по времени)
        pooled = attended.mean(dim=1)
        
        # Предсказания
        rhyme_score = self.rhyme_head(pooled)
        meter_score = self.meter_head(pooled)
        style_probs = self.style_head(pooled)
        
        # Общая оценка качества
        quality_input = torch.cat([pooled, style_probs], dim=1)
        quality_score = self.quality_head(quality_input)
        
        return {
            'rhyme_score': rhyme_score,
            'meter_score': meter_score,
            'style_probs': style_probs,
            'quality_score': quality_score,
            'pooled': pooled
        }

class BERTScorer:
    """Оценка стихов с использованием BERT"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._initialize_bert()
    
    def _initialize_bert(self):
        """Инициализация BERT модели"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers не доступен для BERTScorer")
            return
        
        try:
            from transformers import BertModel, BertTokenizer
            self.tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
            self.model = BertModel.from_pretrained('DeepPavlov/rubert-base-cased')
            self.model.eval()
            print("✅ BERT модель загружена")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки BERT: {e}")
            self.model = None
    
    def score_poem(self, poem: str) -> float:
        """Оценка качества стихотворения"""
        if self.model is None:
            return self._fallback_score(poem)
        
        try:
            inputs = self.tokenizer(
                poem,
                return_tensors='pt',
                truncation=True,
                max_length=128,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Используем embedding [CLS] токена
                features = outputs.last_hidden_state[:, 0, :]
            
            # Простая эвристика для оценки
            score = torch.mean(features).item()
            return self._normalize_score(score)
        except:
            return self._fallback_score(poem)
    
    def _fallback_score(self, poem: str) -> float:
        """Упрощенная оценка при отсутствии BERT"""
        lines = poem.split('\n')
        words = poem.split()
        
        if len(words) < 5:
            return 0.3
        
        score = 0.5
        score += min(len(lines) / 20, 0.3)
        score += min(len(set(words)) / len(words), 0.2)
        
        return min(score, 0.9)
    
    def _normalize_score(self, score: float) -> float:
        """Нормализация оценки"""
        return 1 / (1 + np.exp(-score * 10))

class GPT2PoetryGenerator:
    """Генератор стихов на основе GPT-2"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._initialize_gpt2()
    
    def _initialize_gpt2(self):
        """Инициализация GPT-2 модели"""
        if not TRANSFORMERS_AVAILABLE:
            print("⚠️ Transformers не доступен для GPT2PoetryGenerator")
            return
        
        try:
            from transformers import GPT2LMHeadModel, GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
            self.model.eval()
            print("✅ GPT-2 модель загружена")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки GPT-2: {e}")
            self.model = None
    
    def generate(self, prompt: str, style: str = "лирический", 
                 max_length: int = 100, temperature: float = 0.8) -> str:
        """Генерация стиха"""
        if self.model is None:
            return self._fallback_generate(prompt, style)
        
        try:
            # Добавляем указание стиля
            style_prompt = f"[{style.upper()}] {prompt}"
            
            inputs = self.tokenizer.encode(style_prompt, return_tensors='pt')
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=3
                )
            
            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Убираем промпт из результата
            result = generated[len(style_prompt):].strip()
            
            if not result:
                return self._fallback_generate(prompt, style)
            
            return result
        except:
            return self._fallback_generate(prompt, style)
    
    def _fallback_generate(self, prompt: str, style: str) -> str:
        """Упрощенная генерация при отсутствии GPT-2"""
        templates = {
            "лирический": [
                f"{prompt} в тишине ночной,\nИ сердце бьется под луной.",
                f"О {prompt} я вспоминаю,\nИ душу светом наполняю.",
                f"Как {prompt} прекрасна и чиста,\nНаполнена небес мечтой."
            ],
            "юмористический": [
                f"{prompt} - вот это да!\nСмешно до слез, ей-бога!",
                f"Про {prompt} расскажу я вам,\nЧтоб поднялся настроенье.",
                f"{prompt} такой смешной,\nЧто хохотать готов любой."
            ],
            "философский": [
                f"Что есть {prompt} в мире этом?\nВопрос, терзающий поэта.",
                f"{prompt} - суть бытия,\nИ смысл существования.",
                f"В {prompt} скрыта истина,\nПознания глубина."
            ]
        }
        
        import random
        return random.choice(templates.get(style, templates["лирический"]))