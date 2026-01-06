"""
Ядро поэтического агента с ML и RL
"""

import sys
import datetime
from typing import Dict, List, Tuple, Any
import json
import os

# Проверяем доступность torch
try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch не установлен. ML/RL функции будут отключены.")
    TORCH_AVAILABLE = False

from generator import PoemGenerator
from analyzer import PoemAnalyzer
from reporter import ReportGenerator
from rewriter import PoemRewriter

# Динамический импорт ML/RL модулей
if TORCH_AVAILABLE:
    try:
        from neural_models import PoetryCritic, BERTScorer, GPT2PoetryGenerator
        from rl_agent import ActorCriticAgent, DQNAgent, PoetryEnvironment
        ML_AVAILABLE = True
    except ImportError as e:
        print(f"⚠️ Ошибка импорта ML модулей: {e}")
        ML_AVAILABLE = False
else:
    ML_AVAILABLE = False

class EnhancedPoetryAgentCore:
    """Ядро поэтического агента с ML и RL"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
        # Базовые компоненты
        self.generator = PoemGenerator()
        self.analyzer = PoemAnalyzer()
        self.rewriter = PoemRewriter()
        self.reporter = ReportGenerator()
        
        # ML компоненты
        self.ml_enabled = ML_AVAILABLE
        self.rl_enabled = ML_AVAILABLE
        
        if self.ml_enabled:
            self._initialize_ml_models()
        
        if self.rl_enabled:
            self._initialize_rl_agent()
        
        self.stats = {
            "total_poems": 0,
            "avg_improvement": 0.0,
            "common_issues": [],
            "session_start": datetime.datetime.now(),
            "ml_predictions": 0,
            "rl_improvements": 0,
            "ml_scores": [],
            "rl_rewards": []
        }
        
        if self.verbose:
            print("🤖 Поэтический агент инициализирован")
            if self.ml_enabled:
                print("   ✅ ML модели доступны")
            if self.rl_enabled:
                print("   ✅ RL агент доступен")
    
    def _initialize_ml_models(self):
        """Инициализация ML моделей"""
        try:
            self.bert_scorer = BERTScorer()
            self.gpt_generator = GPT2PoetryGenerator()
            
            # Загружаем предобученного критика если есть
            self.critic = None
            critic_path = "models/poetry_critic.pth"
            if os.path.exists(critic_path):
                try:
                    self.critic = PoetryCritic()
                    self.critic.load_state_dict(torch.load(critic_path))
                    self.critic.eval()
                    print("✅ Нейросетевой критик загружен")
                except:
                    print("⚠️ Не удалось загрузить критика")
            
            print("✅ ML модели инициализированы")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации ML моделей: {e}")
            self.ml_enabled = False
    
    def _initialize_rl_agent(self):
        """Инициализация RL агента"""
        try:
            # Создаем среду и агента
            self.rl_env = PoetryEnvironment()
            state_dim = self.rl_env.max_lines * self.rl_env.max_line_length + 2
            action_dim = self.rl_env.action_space_size
            
            self.rl_agent = ActorCriticAgent(state_dim, action_dim)
            
            # Пытаемся загрузить обученного агента
            agent_path = "models/rl_agent.pth"
            if os.path.exists(agent_path):
                try:
                    self.rl_agent.policy.load_state_dict(torch.load(agent_path))
                    print("✅ RL агент загружен")
                except:
                    print("⚠️ Не удалось загрузить RL агента")
            
            print("✅ RL агент инициализирован")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации RL агента: {e}")
            self.rl_enabled = False
    
    def process_request(self, prompt: str, style: str = "лирический", callback=None) -> Dict:
        """Основной метод обработки запроса с ML и RL"""
        
        def log(msg):
            if self.verbose:
                print(f"🤖 {msg}")
            if callback:
                callback("status", msg)
        
        results = {}
        
        try:
            # Генерация стиха
            log("Генерирую стих..." + (" (с использованием ML)" if self.ml_enabled else ""))
            
            if self.ml_enabled and hasattr(self, 'gpt_generator'):
                # Используем GPT-2 для генерации
                ml_poem = self.gpt_generator.generate(prompt, style)
                
                # Также генерируем классическим методом
                classic_poem, lines = self.generator.generate_poem(prompt, style)
                
                # Оцениваем оба варианта и выбираем лучший
                if self._evaluate_poem_ml(ml_poem) >= self._evaluate_poem_ml(classic_poem):
                    poem = ml_poem
                    log("Выбран вариант от GPT-2")
                else:
                    poem = classic_poem
                    log("Выбран классический вариант")
            else:
                poem, lines = self.generator.generate_poem(prompt, style)
            
            log(f"Сгенерировано стихотворение")
            
            if callback:
                callback("original_poem", poem)
            
            # Анализ стиха
            log("Анализирую по чек-листу..." + (" (с ML оценкой)" if self.ml_enabled else ""))
            
            # Базовый анализ
            original_analysis = self.analyzer.analyze(str(poem), str(prompt), str(style))
            
            # ML анализ если доступен
            if self.ml_enabled:
                ml_score = self._evaluate_poem_ml(poem)
                original_analysis['ml_score'] = ml_score
                self.stats['ml_predictions'] += 1
                self.stats['ml_scores'].append(ml_score)
                
                log(f"ML оценка: {ml_score:.2f}")
            
            if callback:
                callback("analysis", original_analysis)
            
            diagnostics = original_analysis.get("diagnostics", [])
            improvements = []
            
            if diagnostics:
                log(f"Найдено проблем: {len(diagnostics)}. Улучшаю..." + 
                   (" (с использованием RL)" if self.rl_enabled else ""))
                
                if self.rl_enabled:
                    # Используем RL для улучшения
                    improved_poem, rl_improvements, rl_reward = self._improve_with_rl(
                        poem, diagnostics, prompt, style
                    )
                    improvements = rl_improvements
                    
                    if rl_reward > 0:
                        self.stats['rl_improvements'] += 1
                        self.stats['rl_rewards'].append(rl_reward)
                        log(f"RL улучшение применено, награда: {rl_reward:.2f}")
                else:
                    # Классическое улучшение
                    improved_poem, improvements = self.rewriter.rewrite_poem(
                        str(poem), diagnostics, str(prompt), str(style)
                    )
                
                # Анализ улучшенного стиха
                improved_analysis = self.analyzer.analyze(str(improved_poem), str(prompt), str(style))
                
                # ML оценка улучшенного стиха
                if self.ml_enabled:
                    improved_ml_score = self._evaluate_poem_ml(improved_poem)
                    improved_analysis['ml_score'] = improved_ml_score
                
                # Формируем отчет
                report = self.reporter.generate_report(
                    str(poem), str(improved_poem),
                    original_analysis, improved_analysis,
                    improvements, str(prompt), str(style)
                )
                
                self._update_stats(original_analysis, improved_analysis, diagnostics)
                
                results = {
                    "original_poem": poem,
                    "improved_poem": improved_poem,
                    "original_analysis": original_analysis,
                    "improved_analysis": improved_analysis,
                    "improvements": improvements,
                    "report": report,
                    "diagnostics": diagnostics,
                    "score_before": original_analysis.get("score", 0),
                    "score_after": improved_analysis.get("score", 0),
                    "ml_score_before": original_analysis.get("ml_score", 0),
                    "ml_score_after": improved_analysis.get("ml_score", 0),
                    "ml_used": self.ml_enabled,
                    "rl_used": self.rl_enabled,
                    "success": True
                }
            else:
                log("Стих уже хорош, улучшения не требуются!")
                results = {
                    "original_poem": poem,
                    "improved_poem": poem,
                    "original_analysis": original_analysis,
                    "improved_analysis": original_analysis,
                    "improvements": ["Улучшения не требуются"],
                    "report": "Стих соответствует всем критериям с первого раза!",
                    "diagnostics": [],
                    "score_before": original_analysis.get("score", 0),
                    "score_after": original_analysis.get("score", 0),
                    "ml_score_before": original_analysis.get("ml_score", 0) if 'ml_score' in original_analysis else 0,
                    "ml_score_after": original_analysis.get("ml_score", 0) if 'ml_score' in original_analysis else 0,
                    "ml_used": self.ml_enabled,
                    "rl_used": self.rl_enabled,
                    "success": True
                }
            
            self.stats["total_poems"] += 1
            
            if callback:
                callback("result", results)
            
            log("Обработка завершена!")
            
        except Exception as e:
            import traceback
            error_msg = f"Ошибка при обработке: {str(e)}\n{traceback.format_exc()}"
            log(error_msg)
            results = {
                "success": False,
                "error": error_msg,
                "improved_poem": "Извините, произошла ошибка при генерации стиха."
            }
            
            if callback:
                callback("error", error_msg)
        
        return results
    
    def _evaluate_poem_ml(self, poem: str) -> float:
        """Оценка стиха с помощью ML моделей"""
        if not self.ml_enabled:
            return 0.5
        
        try:
            # Используем BERT скоррер
            bert_score = self.bert_scorer.score_poem(poem)
            
            # Используем нейросетевого критика если доступен
            critic_score = 0.5
            if self.critic is not None:
                # Преобразуем стих в тензор (упрощенно)
                tokens = self._poem_to_tokens(poem)
                if tokens is not None:
                    tokens_tensor = torch.LongTensor(tokens).unsqueeze(0)
                    with torch.no_grad():
                        predictions = self.critic(tokens_tensor)
                    critic_score = predictions['quality_score'].item()
            
            # Объединяем оценки
            final_score = 0.6 * bert_score + 0.4 * critic_score
            return final_score
        except:
            return 0.5
    
    def _poem_to_tokens(self, poem: str):
        """Преобразование стиха в токены (упрощенно)"""
        # В реальной реализации нужно использовать токенизатор
        # Здесь упрощенная версия для демонстрации
        words = poem.split()
        tokens = []
        for word in words[:50]:  # Ограничиваем длину
            # Простое преобразование слова в число (хеш)
            token = abs(hash(word)) % 1000
            tokens.append(token)
        
        # Добавляем padding до 50 токенов
        if len(tokens) < 50:
            tokens.extend([0] * (50 - len(tokens)))
        
        return tokens[:50]
    
    def _improve_with_rl(self, poem: str, diagnostics: List[str], 
                        prompt: str, style: str) -> Tuple[str, List[str], float]:
        """Улучшение стиха с помощью RL"""
        if not self.rl_enabled:
            # Возвращаем классическое улучшение
            improved_poem, improvements = self.rewriter.rewrite_poem(
                poem, diagnostics, prompt, style
            )
            return improved_poem, improvements, 0.0
        
        try:
            # Преобразуем стиль в числовой ID
            style_id = {"лирический": 0, "юмористический": 1, "философский": 2}.get(style, 0)
            
            # Сбрасываем среду
            state = self.rl_env.reset(prompt, style_id)
            
            # Создаем базовое улучшение классическим методом
            base_improved, _ = self.rewriter.rewrite_poem(poem, diagnostics, prompt, style)
            
            # Используем RL для дополнительных улучшений
            episode_reward = 0
            max_steps = min(len(diagnostics) * 2, 20)
            
            for step in range(max_steps):
                # Выбираем действие
                action = self.rl_agent.select_action(state, temperature=0.7)
                
                # Выполняем действие
                next_state, reward, done, _ = self.rl_env.step(action)
                episode_reward += reward
                
                # Запоминаем награду для обучения
                self.rl_agent.remember(reward)
                
                # Переходим в следующее состояние
                state = next_state
                
                if done:
                    break
            
            # Получаем RL-улучшенный стих из среды
            rl_poem = self.rl_env.get_poem_text()
            
            # Обновляем политику RL агента
            loss = self.rl_agent.update()
            
            # Смешиваем классическое и RL улучшение
            if len(rl_poem.strip()) > len(poem) // 2:
                # Используем RL стих если он достаточно длинный
                improved_poem = rl_poem
                improvements = [f"RL улучшение (награда: {episode_reward:.2f})"]
                improvements.extend(diagnostics[:3])
            else:
                # Используем классическое улучшение
                improved_poem = base_improved
                improvements = ["Классическое улучшение"]
                improvements.extend(diagnostics[:3])
            
            return improved_poem, improvements, episode_reward
            
        except Exception as e:
            print(f"⚠️ Ошибка RL улучшения: {e}")
            # Возвращаем классическое улучшение
            improved_poem, improvements = self.rewriter.rewrite_poem(
                poem, diagnostics, prompt, style
            )
            return improved_poem, improvements, 0.0
    
    def _update_stats(self, orig: Dict, impr: Dict, issues: List[str]):
        """Обновление статистики"""
        try:
            improvement = impr.get("score", 0) - orig.get("score", 0)
            total = self.stats["total_poems"]
            if total > 0:
                self.stats["avg_improvement"] = (
                    self.stats["avg_improvement"] * (total - 1) + improvement
                ) / total
            else:
                self.stats["avg_improvement"] = improvement
            
            for issue in issues[:3]:
                if isinstance(issue, str):
                    self.stats["common_issues"].append(issue)
        except:
            pass
    
    def get_stats(self) -> Dict:
        """Получить статистику"""
        stats = self.stats.copy()
        stats["session_duration"] = str(datetime.datetime.now() - stats["session_start"])
        
        # Добавляем ML/RL статистику
        stats["ml_enabled"] = self.ml_enabled
        stats["rl_enabled"] = self.rl_enabled
        
        if self.stats['ml_scores']:
            stats["avg_ml_score"] = np.mean(self.stats['ml_scores']) if TORCH_AVAILABLE else 0
        else:
            stats["avg_ml_score"] = 0
        
        if self.stats['rl_rewards']:
            stats["avg_rl_reward"] = np.mean(self.stats['rl_rewards']) if TORCH_AVAILABLE else 0
        else:
            stats["avg_rl_reward"] = 0
        
        return stats
    
    def format_poem_display(self, poem: str, title: str = "") -> str:
        """Форматирование стиха"""
        if isinstance(poem, list):
            poem = "\n".join(poem)
        
        lines = str(poem).split('\n')
        formatted = ""
        if title:
            formatted += f"📖 {title}\n"
        formatted += "─" * 40 + "\n"
        for i, line in enumerate(lines, 1):
            formatted += f"{i:2d}. {line}\n"
        formatted += "─" * 40
        return formatted
    
    def format_analysis_display(self, analysis: Dict) -> str:
        """Форматирование анализа"""
        score = analysis.get("score", 0)
        rhyme = analysis.get("rhyme", {})
        meter = analysis.get("meter", {})
        style = analysis.get("style", {})
        goal = analysis.get("goal", {})
        
        result = "📊 Результаты анализа:\n"
        result += f"Общая оценка: {score:.2f}/1.0\n"
        
        # Добавляем ML оценку если есть
        if 'ml_score' in analysis:
            result += f"ML оценка: {analysis['ml_score']:.2f}/1.0\n"
        
        result += "\n✅ Критерии:\n"
        result += f"  • Рифма: {'✓' if rhyme.get('is_ok') else '✗'} ({rhyme.get('rhyme_type', 'нет')})\n"
        result += f"  • Размер: {'✓' if meter.get('is_ok') else '✗'} ({meter.get('meter', 'не опр.')})\n"
        result += f"  • Стиль: {'✓' if style.get('is_ok') else '✗'} (совпадение: {style.get('style_match', 0):.2f})\n"
        result += f"  • Цель: {'✓' if goal.get('is_ok') else '✗'} (схожесть: {goal.get('similarity', 0):.2f})\n"
        
        diag = analysis.get("diagnostics", [])
        if diag:
            result += "\n⚠️ Проблемы:\n"
            for d in diag[:5]:
                result += f"  • {d}\n"
        
        return result
    
    def train_models(self, episodes: int = 100):
        """Обучение ML и RL моделей"""
        if not TORCH_AVAILABLE:
            return {"error": "PyTorch не установлен"}
        
        try:
            results = {
                "rl_training": {},
                "critic_training": {}
            }
            
            # Обучение RL агента
            if self.rl_enabled:
                print(f"🎯 Обучение RL агента ({episodes} эпизодов)...")
                
                rl_rewards = []
                rl_losses = []
                
                for episode in range(episodes):
                    # Случайный промпт и стиль
                    prompts = ["любовь", "осень", "город", "ночь", "мечта"]
                    styles = [0, 1, 2]  # 0: лирика, 1: юмор, 2: философия
                    
                    prompt = np.random.choice(prompts)
                    style = np.random.choice(styles)
                    
                    # Сброс среды
                    state = self.rl_env.reset(prompt, style)
                    
                    episode_reward = 0
                    done = False
                    
                    while not done:
                        action = self.rl_agent.select_action(state, temperature=0.5)
                        next_state, reward, done, _ = self.rl_env.step(action)
                        self.rl_agent.remember(reward)
                        episode_reward += reward
                        state = next_state
                    
                    # Обновление политики
                    loss = self.rl_agent.update()
                    
                    rl_rewards.append(episode_reward)
                    rl_losses.append(loss)
                    
                    if (episode + 1) % 10 == 0:
                        print(f"  Эпизод {episode+1}: Награда = {episode_reward:.2f}, Потеря = {loss:.4f}")
                
                results["rl_training"] = {
                    "episodes": episodes,
                    "avg_reward": np.mean(rl_rewards) if rl_rewards else 0,
                    "avg_loss": np.mean(rl_losses) if rl_losses else 0,
                    "best_reward": np.max(rl_rewards) if rl_rewards else 0
                }
                
                # Сохраняем обученного агента
                torch.save(self.rl_agent.policy.state_dict(), "models/rl_agent.pth")
                print("✅ RL агент обучен и сохранен")
            
            return results
            
        except Exception as e:
            return {"error": str(e)}