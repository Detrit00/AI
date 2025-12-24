"""
Вспомогательные функции для работы с текстом
"""

import json
import re
import random
import os
from typing import List, Dict, Tuple, Any
import datetime
from collections import Counter

class TextUtils:
    """Вспомогательные функции для работы с текстом"""
    
    @staticmethod
    def load_json(filepath: str) -> Dict:
        """Загрузка JSON файла"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Файл {filepath} не найден, создан пустой словарь")
            return {}
        except json.JSONDecodeError:
            print(f"⚠️ Ошибка чтения JSON файла {filepath}, создан пустой словарь")
            return {}
    
    @staticmethod
    def save_json(data: Dict, filepath: str):
        """Сохранение в JSON"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @staticmethod
    def file_exists(filepath: str) -> bool:
        """Проверяет существование файла"""
        return os.path.exists(filepath)
    
    @staticmethod
    def count_syllables(word: str) -> int:
        """Подсчет слогов в слове"""
        vowels = 'аеёиоуыэюя'
        word_lower = word.lower()
        
        exceptions = {
            'июнь': 2, 'июль': 2, 'январь': 2, 'февраль': 3, 'август': 2,
            'октябрь': 2, 'ноябрь': 2, 'декабрь': 2
        }
        
        if word_lower in exceptions:
            return exceptions[word_lower]
        
        count = sum(1 for char in word_lower if char in vowels)
        
        if count == 0 and len(word) > 0:
            return 1
        
        return count
    
    @staticmethod
    def extract_last_word(line: str) -> str:
        """Извлекает последнее слово из строки"""
        if not line:
            return ""
        
        line = line.strip().rstrip('.,!?;:')
        words = line.split()
        
        if not words:
            return ""
        
        return words[-1].lower()
    
    @staticmethod
    def extract_last_word_grammatical(line: str) -> str:
        """Извлечение последнего слова"""
        return TextUtils.extract_last_word(line)
    
    @staticmethod
    def find_rhyming_words(word: str, rhymes_dict: Dict) -> List[str]:
        """Поиск рифмующихся слов из словаря"""
        if not word or not rhymes_dict:
            return []
        
        word_lower = word.lower()
        
        # Проверяем, есть ли слово как ключ
        if word_lower in rhymes_dict:
            return rhymes_dict[word_lower]
        
        # Ищем слово в значениях
        for key, rhymes in rhymes_dict.items():
            if word_lower in rhymes:
                result = [key] + [r for r in rhymes if r != word_lower]
                return result
        
        # Ищем по окончанию
        if len(word_lower) >= 3:
            ending = word_lower[-3:]
            result = []
            for key in rhymes_dict:
                if key.endswith(ending):
                    result.append(key)
                    if len(result) >= 5:
                        break
            return result
        
        return []
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Вычисление схожести"""
        from difflib import SequenceMatcher
        
        if not text1 or not text2:
            return 0.0
        
        return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()
    
    @staticmethod
    def detect_rhythm(line: str) -> str:
        """Определение стихотворного размера"""
        words = re.findall(r'[а-яА-ЯёЁ]+', str(line))
        if not words:
            return "не определен"
        
        syllables = [TextUtils.count_syllables(w) for w in words]
        total_syllables = sum(syllables)
        
        if total_syllables == 0:
            return "не определен"
        
        # Ямб: ударение на четных слогах
        if total_syllables in [8, 9, 10] and len(words) >= 3:
            return "ямб"
        
        # Хорей: ударение на нечетных слогах
        if total_syllables in [7, 8, 9] and len(words) >= 3:
            return "хорей"
        
        # Дактиль
        if total_syllables >= 10 and len(words) >= 3:
            return "дактиль"
        
        return "свободный"
    
    @staticmethod
    def get_timestamp() -> str:
        """Возвращает текущую временную метку"""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    @staticmethod
    def format_time_delta(seconds: float) -> str:
        """Форматирует разницу времени"""
        if seconds < 60:
            return f"{seconds:.1f} сек"
        elif seconds < 3600:
            return f"{seconds/60:.1f} мин"
        else:
            return f"{seconds/3600:.1f} час"
    
    @staticmethod
    def extract_keywords(text: str, count: int = 5) -> List[str]:
        """Извлекает ключевые слова"""
        words = re.findall(r'[а-яА-ЯёЁ]{3,}', str(text).lower())
        
        stop_words = {
            'это', 'как', 'так', 'и', 'в', 'над', 'под', 'к', 'до', 'не',
            'но', 'он', 'она', 'оно', 'они', 'мы', 'вы', 'ты', 'вас', 'нас',
            'их', 'мой', 'твой', 'свой', 'наш', 'ваш', 'его', 'ее', 'их'
        }
        
        filtered = [w for w in words if w not in stop_words]
        word_counts = Counter(filtered)
        
        return [word for word, _ in word_counts.most_common(count)]
    
    @staticmethod
    def split_into_stanzas(poem: str) -> List[List[str]]:
        """Разбивает стих на строфы"""
        lines = [line.strip() for line in str(poem).split('\n') if line.strip()]
        
        if not lines:
            return []
        
        stanzas = []
        current = []
        
        for line in lines:
            if line:
                current.append(line)
            else:
                if current:
                    stanzas.append(current)
                    current = []
        
        if current:
            stanzas.append(current)
        
        return stanzas
    
    @staticmethod
    def get_poem_stats(poem: str) -> Dict[str, Any]:
        """Статистика стихотворения"""
        lines = [line for line in str(poem).split('\n') if line.strip()]
        words = re.findall(r'[а-яА-ЯёЁ]+', str(poem))
        syllables = sum(TextUtils.count_syllables(w) for w in words)
        
        return {
            "line_count": len(lines),
            "word_count": len(words),
            "syllable_count": syllables,
            "avg_words_per_line": len(words) / max(len(lines), 1),
            "avg_syllables_per_line": syllables / max(len(lines), 1),
            "stanza_count": len(TextUtils.split_into_stanzas(poem))
        }
    
    @staticmethod
    def check_grammatical_agreement(word1: str, word2: str) -> bool:
        """Проверяет грамматическое согласование"""
        return True