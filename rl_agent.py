"""
Reinforcement Learning агент для генерации и улучшения стихов
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Dict, Tuple, Any
import copy

class PolicyNetwork(nn.Module):
    """Политика для RL агента"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.value_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        logits = self.network(state)
        value = self.value_network(state)
        return logits, value

class ActorCriticAgent:
    """Агент с архитектурой Actor-Critic"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = 0.99
        
        self.memory = []
        self.log_probs = []
        self.values = []
        self.rewards = []
    
    def select_action(self, state, temperature=1.0):
        """Выбор действия"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        logits, value = self.policy(state_tensor)
        
        # Применяем temperature для exploration
        probs = F.softmax(logits / temperature, dim=-1)
        
        # Сэмплируем действие
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        
        # Сохраняем для обучения
        self.log_probs.append(action_dist.log_prob(action))
        self.values.append(value)
        
        return action.item()
    
    def update(self):
        """Обновление политики"""
        if not self.rewards:
            return 0.0
        
        rewards = []
        discounted_reward = 0
        
        # Рассчитываем дисконтированные награды
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
        
        # Нормализуем награды
        rewards = torch.FloatTensor(rewards)
        if rewards.std() > 0:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # Рассчитываем потери
        policy_loss = []
        value_loss = []
        
        for log_prob, value, reward in zip(self.log_probs, self.values, rewards):
            advantage = reward - value.item()
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.smooth_l1_loss(value, reward.unsqueeze(0)))
        
        if not policy_loss:
            return 0.0
        
        # Объединяем потери
        policy_loss = torch.stack(policy_loss).sum()
        value_loss = torch.stack(value_loss).sum()
        
        loss = policy_loss + 0.5 * value_loss
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Очищаем память
        self.log_probs = []
        self.values = []
        self.rewards = []
        
        return loss.item()
    
    def remember(self, reward):
        """Запоминание награды"""
        self.rewards.append(reward)

class DQNAgent:
    """Агент Deep Q-Learning"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Основная сеть
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        # Целевая сеть
        self.target_network = copy.deepcopy(self.q_network)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Память для replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # Параметры exploration
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        
        # Счетчик обновлений
        self.update_counter = 0
        self.target_update_freq = 100
    
    def select_action(self, state, training=True):
        """Выбор действия"""
        if training and np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Запоминание перехода"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Обучение на памяти"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        batch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([transition[0] for transition in batch])
        actions = torch.LongTensor([transition[1] for transition in batch]).unsqueeze(1)
        rewards = torch.FloatTensor([transition[2] for transition in batch])
        next_states = torch.FloatTensor([transition[3] for transition in batch])
        dones = torch.FloatTensor([transition[4] for transition in batch])
        
        # Q-values для текущих состояний
        current_q = self.q_network(states).gather(1, actions)
        
        # Q-values для следующих состояний
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Потеря
        loss = self.criterion(current_q.squeeze(), target_q)
        
        # Оптимизация
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Уменьшаем epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Обновляем целевую сеть
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()

class PoetryEnvironment:
    """Среда RL для поэзии"""
    
    def __init__(self, vocab_size=1000, max_lines=8, max_line_length=15):
        self.vocab_size = vocab_size
        self.max_lines = max_lines
        self.max_line_length = max_line_length
        
        # Пространство действий
        self.action_space_size = vocab_size + 5  # слова + специальные действия
        
        self.reset()
    
    def reset(self, prompt="", style=0):
        """Сброс среды"""
        self.current_poem = []
        self.current_line = []
        self.prompt = prompt
        self.style = style
        self.done = False
        self.step_count = 0
        
        return self._get_state()
    
    def step(self, action):
        """Выполнение действия"""
        self.step_count += 1
        reward = 0
        
        # Специальные действия
        if action >= self.vocab_size:
            if action == self.vocab_size:  # Закончить строку
                if self.current_line:
                    self.current_poem.append(self.current_line.copy())
                    self.current_line = []
                    reward = 0.1
            elif action == self.vocab_size + 1:  # Закончить стих
                if len(self.current_poem) >= 2:
                    self.done = True
                    reward = self._calculate_final_reward()
            elif action == self.vocab_size + 2:  # Удалить слово
                if self.current_line:
                    self.current_line.pop()
                    reward = -0.05
            elif action == self.vocab_size + 3:  # Добавить эпитет
                if len(self.current_line) < self.max_line_length:
                    self.current_line.append(self._get_epithet())
                    reward = 0.02
            elif action == self.vocab_size + 4:  # Добавить глагол
                if len(self.current_line) < self.max_line_length:
                    self.current_line.append(self._get_verb())
                    reward = 0.02
        else:
            # Добавить слово
            if len(self.current_line) < self.max_line_length:
                self.current_line.append(action)
                reward = 0.01
        
        # Проверяем условие завершения
        if self.step_count >= 100:
            self.done = True
            reward += self._calculate_final_reward()
        
        return self._get_state(), reward, self.done, {}
    
    def _get_state(self):
        """Получение текущего состояния"""
        # Создаем вектор состояния
        state = np.zeros(self.max_lines * self.max_line_length + 2)
        
        # Заполняем текущий стих
        idx = 0
        for line in self.current_poem:
            for word in line:
                if idx < len(state) - 2:
                    state[idx] = word / self.vocab_size
                    idx += 1
        
        # Добавляем текущую строку
        for word in self.current_line:
            if idx < len(state) - 2:
                state[idx] = word / self.vocab_size
                idx += 1
        
        # Добавляем стиль и количество строк
        state[-2] = self.style / 3
        state[-1] = len(self.current_poem) / self.max_lines
        
        return state
    
    def _calculate_final_reward(self):
        """Расчет финальной награды"""
        if not self.current_poem:
            return -0.5
        
        reward = 0.0
        
        # Награда за количество строк
        line_count = len(self.current_poem)
        if 4 <= line_count <= 8:
            reward += 0.2
        elif line_count > 0:
            reward += 0.1
        
        # Награда за разнообразие слов
        all_words = []
        for line in self.current_poem:
            all_words.extend(line)
        
        if all_words:
            unique_ratio = len(set(all_words)) / len(all_words)
            reward += unique_ratio * 0.1
        
        # Награда за завершение
        reward += 0.3
        
        return reward
    
    def _get_epithet(self):
        """Получение эпитета"""
        epithets = [100, 101, 102, 103, 104]  # ID эпитетов
        return random.choice(epithets)
    
    def _get_verb(self):
        """Получение глагола"""
        verbs = [200, 201, 202, 203, 204]  # ID глаголов
        return random.choice(verbs)
    
    def get_poem_text(self, vocab_dict=None):
        """Получение текста стиха"""
        if vocab_dict is None:
            vocab_dict = {}
        
        lines = []
        for line in self.current_poem:
            line_words = []
            for word_id in line:
                if word_id in vocab_dict:
                    line_words.append(vocab_dict[word_id])
                else:
                    line_words.append(f"word_{word_id}")
            lines.append(" ".join(line_words))
        
        return "\n".join(lines)