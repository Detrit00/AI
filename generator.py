"""
Модуль генерации стихов по запросу
"""

import re
import random
from typing import List, Dict, Tuple
from utils import TextUtils
from grammar_natasha import grammar

class PoemGenerator:
    """Генератор стихов по запросу"""
    
    def __init__(self):
        self.templates = TextUtils.load_json('data/templates.json')
        self.styles = TextUtils.load_json('data/styles.json')
        self.rhymes = TextUtils.load_json('data/rhymes.json')
        
        self.word_bank = self._create_word_bank()
        self.style_words = self._create_style_words()
    
    def _create_word_bank(self):
        """Создает банк слов"""
        return {
            "место": ["река", "лес", "поле", "город", "село", "гора", "озеро", "море", "сад", "дом", "берег"],
            "природа": ["тишина", "луна", "заря", "метель", "вьюга", "звезда", "рассвет", "закат", "туман", "дождь", "ветер"],
            "эмоция": ["печаль", "радость", "тишина", "свет", "грусть", "веселье", "страх", "нежность", "ясность", "тьма"],
            "сущность": ["душа", "песня", "мысль", "мечта", "любовь", "надежда", "вера", "боль", "радость", "грусть", "память"],
            "время": ["ночь", "осень", "зима", "весна", "лето", "утро", "вечер", "день", "сумерки", "рассвет", "миг"],
            "сравнение": ["как свет звезды", "как шепот трав", "как вздох волны", "как полет мечты", "как зыбь тумана"],
            "действие": ["льется", "струится", "мерцает", "тает", "звучит", "поет", "шепчет", "дышит", "светит", "горит", "спит"],
            "животное": ["кот", "пес", "еж", "волк", "заяц", "лиса", "медведь", "сова", "слон", "мышь"],
            "одежда": ["пиджак", "штаны", "галстук", "шляпа", "пальто", "ботинки", "перчатки", "шарф", "платье"],
            "ситуация": ["танцует слон", "поет мышь", "вверх идет дождь", "падает небо", "спит солнце", "молчит ветер"],
            "абстракция": ["жизнь", "любовь", "счастье", "надежда", "вера", "судьба", "память", "время", "смысл"],
            "вопрос": ["Что есть истина?", "Куда идти?", "Зачем жить?", "Что такое счастье?"],
            "цвет": ["белый", "черный", "синий", "красный", "зеленый", "золотой", "серебряный", "багряный", "алый"],
            "звук": ["шепот", "крик", "плач", "смех", "пение", "молчание", "эхо", "тишина", "голос"],
            "наречие": ["тихо", "громко", "быстро", "медленно", "ярко", "темно", "ясно", "светло", "грустно"]
        }
    
    def _create_style_words(self):
        """Создает словари для стилей"""
        return {
            "лирический": {
                "nouns": ["любовь", "душа", "сердце", "мечта", "печаль", "радость", "надежда"],
                "adjectives": ["нежный", "тихий", "светлый", "грустный", "прекрасный", "далекий"],
                "verbs": ["любить", "мечтать", "страдать", "надеяться", "чувствовать", "верить"]
            },
            "юмористический": {
                "nouns": ["кот", "смех", "шутка", "анекдот", "прикол", "курьез", "хохот"],
                "adjectives": ["смешной", "забавный", "нелепый", "уморительный", "странный", "курьезный"],
                "verbs": ["смеяться", "шутить", "хохотать", "улыбаться", "веселиться", "дурачиться"]
            },
            "философский": {
                "nouns": ["жизнь", "смерть", "время", "вечность", "смысл", "бытие", "истина"],
                "adjectives": ["глубокий", "вечный", "сущностный", "абсолютный", "загадочный", "таинственный"],
                "verbs": ["размышлять", "понимать", "сомневаться", "познавать", "осознавать", "существовать"]
            }
        }
    
    def generate_poem(self, prompt: str, style: str = "лирический") -> Tuple[str, List[str]]:
        """Генерация стиха"""
        prompt = str(prompt)
        style = str(style)
        
        prompt_words = re.findall(r'[а-яА-ЯёЁ]{3,}', prompt.lower())
        
        if style not in self.templates:
            style = "лирический"
        
        templates = self.templates.get(style, [])
        if not templates:
            templates = self.templates["лирический"]
        
        # Определяем количество строк
        num_lines = 4 if random.random() > 0.3 else 8
        
        lines = []
        for i in range(num_lines):
            if i < len(templates):
                template_item = templates[i % len(templates)]
                template = template_item.get("template", "")
                target_rhythm = template_item.get("rhythm", "ямб")
            else:
                selected_template = random.choice(templates)
                template = selected_template.get("template", "")
                target_rhythm = selected_template.get("rhythm", "ямб")
            
            if not template:
                template = "[место] [природа]"
            
            line = self._fill_template(template, prompt_words, style, i)
            line = self._adjust_rhythm(line, target_rhythm)
            
            lines.append(line)
        
        # Добавляем рифмы
        if len(lines) >= 2:
            lines = self._add_rhymes(lines, style)
        
        # Добавляем пунктуацию
        lines = self._add_punctuation(lines)
        
        poem = "\n".join(lines)
        return poem, lines
    
    def _fill_template(self, template: str, prompt_words: List[str], 
                      style: str, line_num: int) -> str:
        """Заполняет шаблон"""
        line = template
        
        # Заменяем теги на слова
        for key in self.word_bank:
            tag = f"[{key}]"
            if tag in line:
                # Пробуем использовать слова из запроса
                matching_words = [w for w in prompt_words 
                                if len(w) > 2 and w not in line.lower()]
                if matching_words and random.random() > 0.4:
                    replacement = random.choice(matching_words)
                else:
                    # Используем слова из банка
                    replacement = self._select_word_for_style(key, style)
                
                line = line.replace(tag, replacement, 1)
        
        # Заменяем оставшиеся теги
        while re.search(r'\[.*?\]', line):
            for key in self.word_bank:
                if random.random() > 0.7:
                    replacement = self._select_word_for_style(key, style)
                    line = re.sub(r'\[.*?\]', replacement, line, 1)
                    break
        
        # Первая строка с заглавной буквы
        if line_num == 0 and line:
            line = line[0].upper() + line[1:] if len(line) > 1 else line.upper()
        
        return line
    
    def _select_word_for_style(self, word_type: str, style: str) -> str:
        """Выбирает слово для стиля"""
        if style in self.style_words:
            style_dict = self.style_words[style]
            
            if word_type in ["сущность", "место", "природа", "время", "абстракция"]:
                if style_dict.get("nouns"):
                    return random.choice(style_dict["nouns"])
            elif word_type in ["цвет", "эмоция"]:
                if style_dict.get("adjectives"):
                    return random.choice(style_dict["adjectives"])
            elif word_type in ["действие"]:
                if style_dict.get("verbs"):
                    return random.choice(style_dict["verbs"])
        
        # Если не нашли, берем из общего банка
        if word_type in self.word_bank:
            words = self.word_bank[word_type]
            if words:
                return random.choice(words)
        
        return "слово"
    
    def _adjust_rhythm(self, line: str, target_rhythm: str) -> str:
        """Корректирует ритм"""
        words = line.split()
        if len(words) < 2:
            return line
        
        # Подсчитываем слоги
        syllables = sum(TextUtils.count_syllables(w) for w in words)
        
        # Целевое количество слогов
        target_syllables = {
            "ямб": random.choice([8, 9, 10]),
            "хорей": random.choice([7, 8, 9]),
            "дактиль": random.choice([10, 11, 12]),
        }.get(target_rhythm, 8)
        
        # Корректируем
        diff = target_syllables - syllables
        
        if diff > 0:
            # Добавляем слова
            additions = ["тихо", "ясно", "светло", "грустно", "нежно"]
            for _ in range(min(diff, 2)):
                addition = random.choice(additions)
                insert_pos = random.randint(0, len(words)-1)
                words.insert(insert_pos, addition)
        
        elif diff < 0:
            # Убираем слова
            if len(words) > 3:
                words = words[:max(3, len(words) + diff)]
        
        return " ".join(words)
    
    def _add_rhymes(self, lines: List[str], style: str) -> List[str]:
        """Добавляет рифмы"""
        if len(lines) < 2:
            return lines
        
        # Определяем схему рифмовки
        if len(lines) == 4:
            scheme = "ABAB" if style == "философский" else "AABB"
        elif len(lines) == 8:
            scheme = "ABABCDCD" if random.random() > 0.5 else "AABBCCDD"
        else:
            scheme = "AABB"
        
        # Применяем схему
        try:
            if scheme == "AABB":
                for i in range(0, len(lines)-1, 2):
                    if i+1 < len(lines):
                        lines[i+1] = self._create_rhyme(lines[i], lines[i+1])
            
            elif scheme == "ABAB" and len(lines) >= 4:
                lines[2] = self._create_rhyme(lines[0], lines[2])
                lines[3] = self._create_rhyme(lines[1], lines[3])
            
            elif scheme == "ABABCDCD" and len(lines) >= 8:
                lines[2] = self._create_rhyme(lines[0], lines[2])
                lines[3] = self._create_rhyme(lines[1], lines[3])
                lines[6] = self._create_rhyme(lines[4], lines[6])
                lines[7] = self._create_rhyme(lines[5], lines[7])
        except Exception as e:
            print(f"⚠️ Ошибка при создании рифмы: {e}")
        
        return lines
    
    def _create_rhyme(self, line1: str, line2: str) -> str:
        """Создает рифму"""
        last_word1 = TextUtils.extract_last_word_grammatical(line1)
        if not last_word1 or len(last_word1) < 2:
            return line2
        
        # Ищем рифмующиеся слова
        rhyming_words = TextUtils.find_rhyming_words(last_word1, self.rhymes)
        
        if rhyming_words:
            # Выбираем рифму
            best_rhyme = random.choice(rhyming_words)
            
            # Заменяем последнее слово
            words = line2.split()
            if words:
                words[-1] = best_rhyme
                return " ".join(words)
        
        return line2
    
    def _add_punctuation(self, lines: List[str]) -> List[str]:
        """Добавляет пунктуацию"""
        for i in range(len(lines)):
            line = lines[i].strip()
            
            # Убираем лишние знаки
            while line and line[-1] in ',.!?;:—':
                line = line[:-1]
            
            if i < len(lines) - 1:
                # Не последняя строка
                if random.random() > 0.4:
                    marks = [",", ";", ":", " —"]
                    line += random.choice(marks)
            else:
                # Последняя строка
                if random.random() > 0.5:
                    line += "..."
                else:
                    line += "."
            
            lines[i] = line
        
        return lines