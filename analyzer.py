"""
Модуль анализа стихов по чек-листу
"""

import re
import time
from typing import List, Dict, Tuple
from utils import TextUtils
from grammar_natasha import grammar

class PoemAnalyzer:
    """Анализатор стихов по чек-листу"""
    
    def __init__(self):
        self.styles = TextUtils.load_json('data/styles.json')
        self.rhymes = TextUtils.load_json('data/rhymes.json')
    
    def analyze(self, poem: str, prompt: str, target_style: str) -> Dict:
        """Полный анализ стиха"""
        # Преобразуем poem в строку, если это список
        if isinstance(poem, list):
            poem = "\n".join(poem)
        
        lines = [line.strip() for line in str(poem).split('\n') if line.strip()]
        
        results = {
            "rhyme": self._check_rhyme_with_grammar(lines),
            "meter": self.check_meter(lines),
            "style": self._check_style_with_natasha(str(poem), str(target_style)),
            "goal": self._check_goal_with_natasha(str(poem), str(prompt)),
            "lines": lines,
            "line_count": len(lines),
            "grammar": self._check_grammar(str(poem))
        }
        
        results["score"] = self._calculate_advanced_score(results)
        results["diagnostics"] = self._generate_detailed_diagnostics(results)
        
        return results
    
    def _check_rhyme_with_grammar(self, lines: List[str]) -> Dict:
        """Проверка рифмы"""
        if len(lines) < 2:
            return {
                "is_ok": False, 
                "problems": ["Слишком мало строк для рифмы"], 
                "rhyme_type": "нет", 
                "rhyme_score": 0
            }
        
        last_words = []
        for line in lines:
            word = TextUtils.extract_last_word_grammatical(line)
            if word:
                last_words.append(word)
        
        if len(last_words) < 2:
            return {
                "is_ok": False,
                "problems": ["Не удалось извлечь слова для рифмы"],
                "rhyme_type": "нет",
                "rhyme_score": 0
            }
        
        analyses = []
        for word in last_words:
            analysis = grammar.analyze_word(word)
            if analysis:
                analyses.append(analysis)
            else:
                analyses.append(None)
        
        problems = []
        rhyme_type = ""
        rhyme_score = 0
        good_rhymes = 0
        
        # Проверяем схемы рифмовки
        if len(lines) == 4:
            if (self._words_rhyme_grammatical(last_words[0], last_words[1], analyses[0], analyses[1]) and
                self._words_rhyme_grammatical(last_words[2], last_words[3], analyses[2], analyses[3])):
                rhyme_type = "AABB"
                good_rhymes = 2
            elif (self._words_rhyme_grammatical(last_words[0], last_words[2], analyses[0], analyses[2]) and
                  self._words_rhyme_grammatical(last_words[1], last_words[3], analyses[1], analyses[3])):
                rhyme_type = "ABAB"
                good_rhymes = 2
            else:
                problems.append("Нарушена рифмовка для 4 строк")
        
        elif len(lines) == 8:
            if (self._words_rhyme_grammatical(last_words[0], last_words[1], analyses[0], analyses[1]) and
                self._words_rhyme_grammatical(last_words[2], last_words[3], analyses[2], analyses[3]) and
                self._words_rhyme_grammatical(last_words[4], last_words[5], analyses[4], analyses[5]) and
                self._words_rhyme_grammatical(last_words[6], last_words[7], analyses[6], analyses[7])):
                rhyme_type = "AABBCCDD"
                good_rhymes = 4
            elif (self._words_rhyme_grammatical(last_words[0], last_words[2], analyses[0], analyses[2]) and
                  self._words_rhyme_grammatical(last_words[1], last_words[3], analyses[1], analyses[3]) and
                  self._words_rhyme_grammatical(last_words[4], last_words[6], analyses[4], analyses[6]) and
                  self._words_rhyme_grammatical(last_words[5], last_words[7], analyses[5], analyses[7])):
                rhyme_type = "ABABCDCD"
                good_rhymes = 4
            else:
                problems.append("Нарушена рифмовка для 8 строк")
        
        # Вычисляем оценку рифмы
        total_possible_pairs = len(lines) // 2
        if total_possible_pairs > 0:
            rhyme_score = good_rhymes / total_possible_pairs
        
        if rhyme_score < 0.5:
            problems.append(f"Недостаточно рифм (качество: {rhyme_score:.2f})")
        
        return {
            "is_ok": len(problems) == 0 and rhyme_score > 0.5,
            "problems": problems,
            "rhyme_type": rhyme_type if rhyme_type else "не определен",
            "rhyme_score": round(rhyme_score, 2)
        }
    
    def _words_rhyme_grammatical(self, word1: str, word2: str, 
                                analysis1: Dict, analysis2: Dict) -> bool:
        """Проверяет рифму"""
        if not word1 or not word2:
            return False
        
        return grammar.words_rhyme(word1, word2)
    
    def _check_style_with_natasha(self, poem: str, target_style: str) -> Dict:
        """Проверка соответствия стилю"""
        # Убедимся, что poem - строка
        if isinstance(poem, list):
            poem = "\n".join(poem)
        
        poem = str(poem)
        target_style = str(target_style)
        
        if target_style not in self.styles:
            return {
                "is_ok": False, 
                "problems": [f"Неизвестный стиль: {target_style}"], 
                "style_match": 0
            }
        
        style_info = self.styles[target_style]
        poem_lower = poem.lower()
        
        # Используем простую токенизацию
        tokens = re.findall(r'[а-яА-ЯёЁ]+', poem_lower)
        
        keywords = style_info.get("keywords", [])
        typical_words = style_info.get("typical_words", [])
        
        found_keywords = []
        found_typical = []
        
        for token in tokens:
            word = token.lower()
            
            if word in keywords:
                found_keywords.append(word)
            
            if word in typical_words:
                found_typical.append(word)
        
        problems = []
        match_score = 0
        
        # Проверяем наличие ключевых слов
        min_keywords = 2 if target_style == "лирический" else 1
        if len(found_keywords) < min_keywords:
            problems.append(
                f"Мало ключевых слов стиля '{target_style}' "
                f"(найдено: {len(found_keywords)}, нужно: {min_keywords})"
            )
        
        # Проверяем типичные слова
        if len(found_typical) < 1 and typical_words:
            problems.append(f"Нет типичных слов для стиля '{target_style}'")
        
        # Вычисляем оценку
        total_style_words = len(set(found_keywords + found_typical))
        total_words = len(tokens)
        
        if total_words > 0:
            word_match = total_style_words / min(total_words, 10)
            match_score = word_match
        
        # Бонус за соответствие стилю
        if target_style == "лирический" and any(word in poem_lower for word in ["любовь", "сердце", "душа"]):
            match_score = min(match_score + 0.2, 1.0)
        
        return {
            "is_ok": len(problems) == 0 and match_score > 0.3,
            "problems": problems,
            "style_match": round(match_score, 2),
            "found_keywords": found_keywords[:5]
        }
    
    def _check_goal_with_natasha(self, poem: str, prompt: str) -> Dict:
        """Проверка соответствия цели"""
        # Убедимся, что poem и prompt - строки
        if isinstance(poem, list):
            poem = "\n".join(poem)
        
        poem = str(poem)
        prompt = str(prompt)
        
        from difflib import SequenceMatcher
        
        similarity = SequenceMatcher(None, poem.lower(), prompt.lower()).ratio()
        
        problems = []
        
        # Анализируем слова
        prompt_words = set(re.findall(r'[а-яА-ЯёЁ]{3,}', prompt.lower()))
        poem_words = set(re.findall(r'[а-яА-ЯёЁ]{3,}', poem.lower()))
        
        common_words = prompt_words.intersection(poem_words)
        
        if similarity < 0.1 and len(prompt_words) > 0:
            problems.append(f"Стих не соответствует теме запроса (схожесть: {similarity:.2f})")
        
        if len(common_words) < 1 and len(prompt_words) > 0:
            problems.append("Нет общих слов с запросом")
        
        thematic_score = len(common_words) / max(len(prompt_words), 1)
        
        if thematic_score < 0.2:
            problems.append("Тематическая связность с запросом низкая")
        
        return {
            "is_ok": len(problems) == 0,
            "problems": problems,
            "similarity": round(similarity, 2),
            "common_words": list(common_words)[:5],
            "thematic_score": round(thematic_score, 2)
        }
    
    def _check_grammar(self, poem: str) -> Dict:
        """Проверка грамматики"""
        # Убедимся, что poem - строка
        if isinstance(poem, list):
            poem = "\n".join(poem)
        
        poem = str(poem)
        
        # Простая проверка
        grammar_errors = []
        
        # Проверяем, что стих не пустой
        if not poem.strip():
            grammar_errors.append("Стих пустой")
        
        # Проверяем наличие знаков препинания
        lines = poem.split('\n')
        for i, line in enumerate(lines):
            if line.strip() and not line.strip()[-1] in '.!?,;:…':
                if i == len(lines) - 1:  # Последняя строка
                    grammar_errors.append(f"Строка {i+1}: отсутствует знак препинания в конце")
        
        grammar_score = 1.0 - min(len(grammar_errors) / max(len(lines), 1), 1.0)
        
        return {
            "errors": grammar_errors[:3],
            "score": round(grammar_score, 2),
            "total_errors": len(grammar_errors),
            "is_ok": len(grammar_errors) == 0
        }
    
    def _calculate_advanced_score(self, results: Dict) -> float:
        """Вычисление общей оценки"""
        weights = {
            "rhyme": 0.25,
            "meter": 0.20,
            "style": 0.20,
            "goal": 0.20,
            "grammar": 0.15
        }
        
        score = 0
        
        # Оценка рифмы
        rhyme_data = results["rhyme"]
        if rhyme_data["is_ok"]:
            score += weights["rhyme"]
        else:
            score += rhyme_data.get("rhyme_score", 0) * weights["rhyme"]
        
        # Оценка размера
        meter_data = results["meter"]
        if meter_data["is_ok"]:
            score += weights["meter"]
        else:
            score += 0.5 * weights["meter"]
        
        # Оценка стиля
        style_data = results["style"]
        score += style_data.get("style_match", 0) * weights["style"]
        
        # Оценка цели
        goal_data = results["goal"]
        if goal_data["is_ok"]:
            score += weights["goal"]
        else:
            score += goal_data.get("thematic_score", 0) * weights["goal"]
        
        # Оценка грамматики
        grammar_data = results["grammar"]
        score += grammar_data.get("score", 0) * weights["grammar"]
        
        # Бонус за количество строк
        line_count = results.get("line_count", 0)
        if line_count >= 4:
            score = min(score * 1.1, 1.0)
        
        return round(score, 2)
    
    def _generate_detailed_diagnostics(self, results: Dict) -> List[str]:
        """Генерация списка проблем"""
        diagnostics = []
        
        for key in ["rhyme", "meter", "style", "goal", "grammar"]:
            if key in results:
                problems = results[key].get("problems", [])
                if isinstance(problems, list):
                    diagnostics.extend(problems)
        
        return diagnostics[:10]
    
    def check_meter(self, lines: List[str]) -> Dict:
        """Проверка стихотворного размера"""
        if not lines:
            return {
                "is_ok": False, 
                "problems": ["Нет строк"], 
                "meter": "не определен"
            }
        
        meters = []
        for line in lines:
            meter = TextUtils.detect_rhythm(line)
            meters.append(meter)
        
        problems = []
        main_meter = meters[0] if meters else "не определен"
        
        for i, meter in enumerate(meters[1:], 1):
            if meter != main_meter:
                problems.append(f"Строка {i+1}: нарушен размер ({meter} вместо {main_meter})")
        
        syllable_counts = []
        for line in lines:
            count = TextUtils.count_syllables(line)
            syllable_counts.append(count)
        
        if syllable_counts:
            avg_syllables = sum(syllable_counts) / len(syllable_counts)
            
            for i, count in enumerate(syllable_counts):
                if abs(count - avg_syllables) > 2:
                    problems.append(f"Строка {i+1}: нестабильное количество слогов ({count})")
        else:
            avg_syllables = 0
        
        return {
            "is_ok": len(problems) == 0,
            "problems": problems,
            "meter": main_meter,
            "syllable_counts": syllable_counts,
            "avg_syllables": round(avg_syllables, 1)
        }
    
    def analyze_with_progress(self, poem: str, prompt: str, target_style: str, callback=None) -> Dict:
        """Анализ с обратными вызовами"""
        steps = ["Проверка рифмы", "Проверка размера", "Проверка стиля", 
                 "Проверка цели", "Проверка грамматики"]
        
        for i, step in enumerate(steps, 1):
            if callback:
                callback("status", f"{step}... ({i}/{len(steps)})")
            time.sleep(0.1)
        
        return self.analyze(poem, prompt, target_style)