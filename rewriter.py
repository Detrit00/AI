"""
Модуль переписывания слабых строк стиха с использованием Natasha
"""

import re
import random
from typing import List, Dict, Tuple, Optional
from utils import TextUtils
from grammar_natasha import grammar

class PoemRewriter:
    """Переписыватель слабых строк с использованием Natasha"""
    
    def __init__(self):
        self.rhymes = TextUtils.load_json('data/rhymes.json')
        self.styles = TextUtils.load_json('data/styles.json')
        
        # Загружаем синонимы
        try:
            self.synonyms = TextUtils.load_json('data/synonyms.json')
        except:
            self.synonyms = self._create_default_synonyms()
            TextUtils.save_json(self.synonyms, 'data/synonyms.json')
        
        # Словари для улучшения стиля
        self.style_enhancements = {
            "лирический": {
                "epithets": ["тихий", "нежный", "светлый", "грустный", "прекрасный", "далекий", "милый", "родной"],
                "comparisons": ["как свет звезды", "как шепот трав", "как вздох волны", "как полет мечты"],
                "metaphors": [("ночь", "черное покрывало"), ("любовь", "яркое пламя"), ("жизнь", "короткая свеча")],
                "adverbs": ["тихо", "нежно", "грустно", "светло", "печально", "медленно"]
            },
            "юмористический": {
                "epithets": ["смешной", "забавный", "нелепый", "уморительный", "странный", "курьезный"],
                "comparisons": ["как слон в посудной лавке", "как рыба об лед", "как курица лапой"],
                "hyperboles": ["до небес", "до упаду", "без конца", "через край", "на весь мир"],
                "adverbs": ["смешно", "забавно", "странно", "курьезно", "нелепо"]
            },
            "философский": {
                "epithets": ["глубокий", "вечный", "сущностный", "абсолютный", "загадочный", "таинственный"],
                "questions": [" - вот в чем вопрос", ", но что есть истина?", "... а смысл?", ", вечная загадка"],
                "abstractions": ["бытие", "сущность", "познание", "вечность", "истина", "смысл"],
                "adverbs": ["глубоко", "вечно", "сущностно", "абсолютно", "истинно"]
            }
        }
        
        # Глаголы для разных стилей
        self.verbs_by_style = {
            "лирический": ["любить", "мечтать", "грустить", "вспоминать", "желать", "страдать", "чувствовать"],
            "юмористический": ["смеяться", "шутить", "прыгать", "бегать", "кричать", "улыбаться", "хитрить"],
            "философский": ["размышлять", "понимать", "сомневаться", "познавать", "осознавать", "существовать"]
        }
        
        # Словарь синонимов для обычных слов
        self.common_synonyms = {
            "идти": ["шагать", "ступать", "двигаться", "шествовать"],
            "стоять": ["находиться", "располагаться", "возвышаться"],
            "сидеть": ["восседать", "покоиться", "находиться"],
            "лежать": ["покоиться", "находиться", "располагаться"],
            "говорить": ["шептать", "вещать", "произносить", "изрекать"],
            "видеть": ["зрить", "созерцать", "наблюдать", "узревать"],
            "знать": ["ведать", "понимать", "осознавать", "чувствовать"],
            "хотеть": ["желать", "стремиться", "мечтать", "жаждать"],
            "любить": ["обожать", "боготворить", "предпочитать", "ценить"]
        }
    
    def _create_default_synonyms(self):
        """Создание словаря синонимов по умолчанию"""
        return {
            "хороший": ["прекрасный", "отличный", "замечательный", "чудесный", "великолепный"],
            "плохой": ["ужасный", "скверный", "отвратительный", "неприятный"],
            "большой": ["огромный", "громадный", "колоссальный", "великий"],
            "маленький": ["крошечный", "малюсенький", "миниатюрный", "небольшой"],
            "быстрый": ["скорый", "стремительный", "проворный", "шустрый"],
            "медленный": ["неторопливый", "размеренный", "замедленный"],
            "красивый": ["прекрасный", "изящный", "очаровательный", "привлекательный"],
            "грустный": ["печальный", "унылый", "тоскливый", "скорбный"],
            "веселый": ["радостный", "счастливый", "ликующий", "жизнерадостный"],
            "тихий": ["безмолвный", "неслышный", "спокойный", "бесшумный"],
            "громкий": ["шумный", "звучный", "оглушительный", "резкий"]
        }
    
    def rewrite_poem(self, poem: str, diagnostics: List[str], 
                    prompt: str, target_style: str) -> Tuple[str, List[str]]:
        """Переписывает стих на основе диагностики с использованием Natasha"""
        lines = [line.strip() for line in poem.split('\n') if line.strip()]
        improvements = []
        
        if not lines:
            return poem, ["Стих пустой"]
        
        # Анализируем каждую строку с помощью Natasha
        line_analyses = []
        for i, line in enumerate(lines):
            words = line.split()
            word_analyses = [grammar.analyze_word(w) for w in words]
            line_analyses.append({
                "index": i,
                "text": line,
                "words": words,
                "analyses": word_analyses,
                "is_weak": self._is_weak_line_analyzed(line, word_analyses)
            })
        
        # Классифицируем проблемы
        rhyme_issues = [d for d in diagnostics if "рифм" in d.lower()]
        meter_issues = [d for d in diagnostics if "размер" in d.lower() or "слогов" in d.lower()]
        style_issues = [d for d in diagnostics if "стиль" in d.lower() or "ключевых слов" in d.lower()]
        goal_issues = [d for d in diagnostics if "запрос" in d.lower() or "теме" in d.lower()]
        grammar_issues = [d for d in diagnostics if "граммати" in d.lower() or "согласован" in d.lower()]
        
        # Улучшаем рифмы (приоритетная задача)
        if rhyme_issues:
            lines, rhyme_improvements = self._improve_rhymes_with_grammar(lines, line_analyses, target_style)
            improvements.extend(rhyme_improvements)
            # Обновляем анализы после изменения строк
            line_analyses = self._reanalyze_lines(lines)
        
        # Улучшаем размер
        if meter_issues:
            lines, meter_improvements = self._improve_meter_with_grammar(lines, line_analyses)
            improvements.extend(meter_improvements)
            line_analyses = self._reanalyze_lines(lines)
        
        # Улучшаем стиль
        if style_issues:
            lines, style_improvements = self._improve_style_with_natasha(lines, target_style, prompt)
            improvements.extend(style_improvements)
            line_analyses = self._reanalyze_lines(lines)
        
        # Улучшаем соответствие запросу
        if goal_issues:
            lines, goal_improvements = self._improve_relevance_with_natasha(lines, prompt)
            improvements.extend(goal_improvements)
            line_analyses = self._reanalyze_lines(lines)
        
        # Исправляем грамматические ошибки
        if grammar_issues:
            lines, grammar_improvements = self._fix_grammar_errors(lines, line_analyses)
            improvements.extend(grammar_improvements)
            line_analyses = self._reanalyze_lines(lines)
        
        # Улучшаем слабые строки
        for i, analysis in enumerate(line_analyses):
            if analysis["is_weak"]:
                new_line = self._enhance_line_with_grammar(analysis, target_style)
                if new_line != lines[i]:
                    lines[i] = new_line
                    improvements.append(f"Строка {i+1}: усилена грамматически")
                    line_analyses = self._reanalyze_lines(lines)
        
        # Проверяем и улучшаем рифмы еще раз после всех изменений
        if len(lines) >= 2:
            lines, final_rhyme_improvements = self._final_rhyme_check(lines, target_style)
            improvements.extend(final_rhyme_improvements)
        
        # Обеспечиваем грамматическую целостность
        lines = self._ensure_grammatical_consistency(lines)
        
        # Улучшаем пунктуацию
        lines = self._improve_punctuation(lines)
        
        improved_poem = "\n".join(lines)
        return improved_poem, improvements
    
    def _reanalyze_lines(self, lines: List[str]) -> List[Dict]:
        """Переанализирует строки после изменений"""
        line_analyses = []
        for i, line in enumerate(lines):
            words = line.split()
            word_analyses = [grammar.analyze_word(w) for w in words]
            line_analyses.append({
                "index": i,
                "text": line,
                "words": words,
                "analyses": word_analyses,
                "is_weak": self._is_weak_line_analyzed(line, word_analyses)
            })
        return line_analyses
    
    def _is_weak_line_analyzed(self, line: str, word_analyses: List[Optional[Dict]]) -> bool:
        """Определяет, является ли строка слабой (с использованием анализа Natasha)"""
        words = line.split()
        
        if len(words) < 2:
            return True
        
        if len(words) > 10:
            return True
        
        # Проверяем наличие анализов
        if not any(word_analyses):
            return True
        
        # Проверяем разнообразие частей речи
        pos_tags = []
        for analysis in word_analyses:
            if analysis:
                pos_tags.append(analysis.get('pos', 'X'))
        
        unique_pos = len(set(pos_tags))
        if unique_pos < 2 and len(words) >= 4:
            return True
        
        # Проверяем наличие стоп-слов
        stop_words = {"это", "как", "так", "и", "в", "на", "с", "по", "у", "о", "за"}
        stop_count = sum(1 for w in words if w.lower() in stop_words)
        if stop_count > len(words) / 3:
            return True
        
        # Проверяем согласованность слов
        grammar_errors = 0
        for i in range(1, len(words)):
            if i < len(word_analyses):
                analysis1 = word_analyses[i-1]
                analysis2 = word_analyses[i]
                
                if analysis1 and analysis2:
                    # Проверяем согласование прилагательного и существительного
                    if (analysis1.get('pos') == 'ADJ' and analysis2.get('pos') == 'NOUN'):
                        if not self._check_adjective_noun_agreement(analysis1, analysis2):
                            grammar_errors += 1
        
        if grammar_errors > len(words) / 4:
            return True
        
        return False
    
    def _check_adjective_noun_agreement(self, adj_analysis: Dict, noun_analysis: Dict) -> bool:
        """Проверяет согласование прилагательного с существительным"""
        adj_gender = adj_analysis.get('gender')
        noun_gender = noun_analysis.get('gender')
        adj_case = adj_analysis.get('case')
        noun_case = noun_analysis.get('case')
        adj_number = adj_analysis.get('number')
        noun_number = noun_analysis.get('number')
        
        # Если есть информация о роде, проверяем совпадение
        if adj_gender and noun_gender and adj_gender != noun_gender:
            return False
        
        # Проверяем падеж
        if adj_case and noun_case and adj_case != noun_case:
            return False
        
        # Проверяем число
        if adj_number and noun_number and adj_number != noun_number:
            return False
        
        return True
    
    def _improve_rhymes_with_grammar(self, lines: List[str], analyses: List[Dict], 
                                   style: str) -> Tuple[List[str], List[str]]:
        """Улучшает рифмы с учетом грамматики Natasha"""
        improvements = []
        
        if len(lines) < 2:
            return lines, improvements
        
        # Определяем оптимальную схему рифмовки
        if len(lines) == 4:
            # Для 4 строк
            if style == "философский":
                scheme = "ABAB"
            elif style == "юмористический":
                scheme = "AABB"
            else:
                scheme = random.choice(["AABB", "ABAB", "ABBA"])
        elif len(lines) == 8:
            # Для 8 строк
            schemes = ["ABABCDCD", "AABBCCDD", "ABABABAB", "ABBAABBA"]
            scheme = random.choice(schemes)
        else:
            scheme = "AABB"
        
        improvements.append(f"Применена схема рифмовки: {scheme}")
        
        if scheme == "AABB":
            for i in range(0, len(lines)-1, 2):
                if i+1 < len(lines):
                    lines[i+1], improved = self._create_grammatical_rhyme_pair(
                        lines[i], lines[i+1], analyses[i], analyses[i+1], style
                    )
                    if improved:
                        improvements.append(f"Создана рифма для строк {i+1}-{i+2}")
        
        elif scheme == "ABAB" and len(lines) >= 4:
            # Рифмуем строки 1-3 и 2-4
            lines[2], improved1 = self._create_grammatical_rhyme_pair(
                lines[0], lines[2], analyses[0], analyses[2], style
            )
            lines[3], improved2 = self._create_grammatical_rhyme_pair(
                lines[1], lines[3], analyses[1], analyses[3], style
            )
            
            if improved1 or improved2:
                improvements.append("Созданы перекрестные рифмы ABAB")
        
        elif scheme == "ABBA" and len(lines) >= 4:
            # Рифмуем строки 1-4 и 2-3
            lines[3], improved1 = self._create_grammatical_rhyme_pair(
                lines[0], lines[3], analyses[0], analyses[3], style
            )
            lines[2], improved2 = self._create_grammatical_rhyme_pair(
                lines[1], lines[2], analyses[1], analyses[2], style
            )
            
            if improved1 or improved2:
                improvements.append("Созданы охватные рифмы ABBA")
        
        elif scheme == "ABABCDCD" and len(lines) >= 8:
            # Рифмуем строки 1-3, 2-4, 5-7, 6-8
            lines[2], improved1 = self._create_grammatical_rhyme_pair(
                lines[0], lines[2], analyses[0], analyses[2], style
            )
            lines[3], improved2 = self._create_grammatical_rhyme_pair(
                lines[1], lines[3], analyses[1], analyses[3], style
            )
            lines[6], improved3 = self._create_grammatical_rhyme_pair(
                lines[4], lines[6], analyses[4], analyses[6], style
            )
            lines[7], improved4 = self._create_grammatical_rhyme_pair(
                lines[5], lines[7], analyses[5], analyses[7], style
            )
            
            if improved1 or improved2 or improved3 or improved4:
                improvements.append("Созданы рифмы по схеме ABABCDCD")
        
        return lines, improvements
    
    def _create_grammatical_rhyme_pair(self, line1: str, line2: str, 
                                     analysis1: Dict, analysis2: Dict,
                                     style: str) -> Tuple[str, bool]:
        """Создает грамматически корректную рифмическую пару"""
        # Извлекаем последние слова
        words1 = analysis1["words"]
        words2 = analysis2["words"]
        
        if not words1 or not words2:
            return line2, False
        
        last_word1 = words1[-1]
        last_word2 = words2[-1]
        
        # Анализируем последние слова
        analysis1_word = grammar.analyze_word(last_word1)
        analysis2_word = grammar.analyze_word(last_word2)
        
        if not analysis1_word or not analysis2_word:
            # Если не удалось проанализировать, пробуем простую рифму
            if grammar.words_rhyme(last_word1, last_word2, strict=False):
                return line2, True
            return line2, False
        
        # Проверяем, рифмуются ли слова
        if grammar.words_rhyme(last_word1, last_word2, strict=True):
            # Проверяем грамматическую совместимость
            if self._are_words_grammatically_compatible(analysis1_word, analysis2_word):
                return line2, True
        
        # Если не рифмуются или не совместимы, ищем замену
        rhyming_words = self._find_grammatical_rhymes(last_word1, analysis1_word, style)
        
        if rhyming_words:
            # Выбираем лучшую рифму
            best_rhyme = self._select_best_grammatical_rhyme(
                last_word1, analysis1_word, rhyming_words, line2
            )
            
            if best_rhyme:
                # Приводим рифму к нужной грамматической форме
                target_case = analysis1_word.get('case', 'nomn')
                target_number = analysis1_word.get('number', 'sing')
                
                inflected_rhyme = grammar.inflect_word(best_rhyme, target_case, target_number)
                
                # Заменяем последнее слово
                new_words = words2.copy()
                new_words[-1] = inflected_rhyme
                return " ".join(new_words), True
        
        # Если не нашли подходящую рифму, улучшаем строку другими способами
        return self._enhance_line_for_rhyming(line1, line2, analysis1, analysis2, style), True
    
    def _find_grammatical_rhymes(self, word: str, analysis: Dict, style: str) -> List[str]:
        """Ищет грамматически совместимые рифмы"""
        rhymes = []
        
        # Ищем в словаре рифм
        dict_rhymes = TextUtils.find_rhyming_words(word, self.rhymes)
        if dict_rhymes:
            # Фильтруем по грамматической совместимости
            for rhyme in dict_rhymes:
                if rhyme.lower() == word.lower():
                    continue
                    
                rhyme_analysis = grammar.analyze_word(rhyme)
                if rhyme_analysis and self._are_words_grammatically_compatible(analysis, rhyme_analysis):
                    rhymes.append(rhyme)
        
        # Ищем в синонимах
        normal_form = analysis.get('normal', word.lower())
        if normal_form in self.synonyms:
            for synonym in self.synonyms[normal_form]:
                if synonym.lower() == word.lower():
                    continue
                    
                if grammar.words_rhyme(word, synonym):
                    synonym_analysis = grammar.analyze_word(synonym)
                    if synonym_analysis and self._are_words_grammatically_compatible(analysis, synonym_analysis):
                        rhymes.append(synonym)
        
        # Ищем в словаре обычных синонимов
        if word.lower() in self.common_synonyms:
            for synonym in self.common_synonyms[word.lower()]:
                if grammar.words_rhyme(word, synonym):
                    synonym_analysis = grammar.analyze_word(synonym)
                    if synonym_analysis and self._are_words_grammatically_compatible(analysis, synonym_analysis):
                        rhymes.append(synonym)
        
        # Добавляем слова из стилевых улучшений
        if style in self.style_enhancements:
            style_data = self.style_enhancements[style]
            
            # Проверяем эпитеты
            for epithet in style_data.get("epithets", []):
                if epithet.lower() == word.lower():
                    continue
                    
                if grammar.words_rhyme(word, epithet):
                    epithet_analysis = grammar.analyze_word(epithet)
                    if epithet_analysis and self._are_words_grammatically_compatible(analysis, epithet_analysis):
                        rhymes.append(epithet)
            
            # Проверяем абстракции для философского стиля
            for abstraction in style_data.get("abstractions", []):
                if abstraction.lower() == word.lower():
                    continue
                    
                if grammar.words_rhyme(word, abstraction):
                    abstraction_analysis = grammar.analyze_word(abstraction)
                    if abstraction_analysis and self._are_words_grammatically_compatible(analysis, abstraction_analysis):
                        rhymes.append(abstraction)
        
        # Убираем дубликаты
        return list(set(rhymes))
    
    def _are_words_grammatically_compatible(self, analysis1: Dict, analysis2: Dict) -> bool:
        """Проверяет грамматическую совместимость двух слов"""
        if not analysis1 or not analysis2:
            return True  # Если нет анализа, считаем совместимыми
        
        # Разные части речи обычно рифмуются хуже
        pos1 = analysis1.get('pos')
        pos2 = analysis2.get('pos')
        
        if pos1 and pos2 and pos1 != pos2:
            # Но допускаем рифмы между некоторыми частями речи
            compatible_pos_pairs = {
                ('NOUN', 'NOUN'): True,
                ('ADJ', 'ADJ'): True,
                ('VERB', 'VERB'): True,
                ('NOUN', 'ADJ'): False,  # Существительное и прилагательное обычно не рифмуются
                ('ADJ', 'NOUN'): False,
                ('VERB', 'NOUN'): True,  # Глагол и существительное иногда рифмуются
                ('NOUN', 'VERB'): True
            }
            
            if not compatible_pos_pairs.get((pos1, pos2), False):
                return False
        
        # Для существительных проверяем падеж и число
        if pos1 == 'NOUN' and pos2 == 'NOUN':
            case1 = analysis1.get('case')
            case2 = analysis2.get('case')
            num1 = analysis1.get('number')
            num2 = analysis2.get('number')
            
            # Рифма лучше, если падеж и число совпадают
            if case1 and case2 and case1 != case2:
                return False
            if num1 and num2 and num1 != num2:
                return False
        
        # Для прилагательных проверяем род, падеж и число
        elif pos1 == 'ADJ' and pos2 == 'ADJ':
            gender1 = analysis1.get('gender')
            gender2 = analysis2.get('gender')
            case1 = analysis1.get('case')
            case2 = analysis2.get('case')
            num1 = analysis1.get('number')
            num2 = analysis2.get('number')
            
            if gender1 and gender2 and gender1 != gender2:
                return False
            if case1 and case2 and case1 != case2:
                return False
            if num1 and num2 and num1 != num2:
                return False
        
        return True
    
    def _select_best_grammatical_rhyme(self, base_word: str, base_analysis: Dict,
                                     rhymes: List[str], context_line: str) -> Optional[str]:
        """Выбирает лучшую грамматически совместимую рифму"""
        if not rhymes:
            return None
        
        # Исключаем слова, которые уже есть в строке
        context_words = set(context_line.lower().split())
        available_rhymes = [r for r in rhymes if r.lower() not in context_words]
        
        if not available_rhymes:
            return rhymes[0] if rhymes else None
        
        # Оцениваем каждую рифму
        scored_rhymes = []
        for rhyme in available_rhymes:
            score = 0
            
            # Анализируем рифму
            rhyme_analysis = grammar.analyze_word(rhyme)
            if rhyme_analysis:
                # Полное грамматическое совпадение
                if self._are_words_grammatically_compatible(base_analysis, rhyme_analysis):
                    score += 3
                
                # Совпадение конкретных признаков
                if (rhyme_analysis.get('case') and 
                    base_analysis.get('case') and
                    rhyme_analysis['case'] == base_analysis['case']):
                    score += 2
                
                if (rhyme_analysis.get('number') and 
                    base_analysis.get('number') and
                    rhyme_analysis['number'] == base_analysis['number']):
                    score += 2
                
                if (rhyme_analysis.get('gender') and 
                    base_analysis.get('gender') and
                    rhyme_analysis['gender'] == base_analysis['gender']):
                    score += 2
            
            # Фонетическое качество
            if grammar.words_rhyme(base_word, rhyme, strict=True):
                score += 2
            
            # Длина слова
            if abs(len(rhyme) - len(base_word)) <= 2:
                score += 1
            
            scored_rhymes.append((score, rhyme))
        
        if not scored_rhymes:
            return available_rhymes[0] if available_rhymes else None
        
        scored_rhymes.sort(reverse=True)
        return scored_rhymes[0][1] if scored_rhymes else available_rhymes[0]
    
    def _enhance_line_for_rhyming(self, line1: str, line2: str, 
                                analysis1: Dict, analysis2: Dict,
                                style: str) -> str:
        """Улучшает строку для создания рифмы"""
        words2 = analysis2["words"].copy()
        
        if len(words2) < 2:
            return line2
        
        # Стратегии улучшения в зависимости от стиля
        if style == "лирический":
            strategies = [
                self._add_lyrical_epithet,
                self._change_to_lyrical_word,
                self._add_adverb_to_line,
                self._reorder_for_lyrical_flow
            ]
        elif style == "юмористический":
            strategies = [
                self._add_humorous_element,
                self._change_to_humorous_word,
                self._add_exaggeration,
                self._reorder_for_humorous_effect
            ]
        else:  # философский
            strategies = [
                self._add_philosophical_depth,
                self._change_to_philosophical_word,
                self._add_profound_adverb,
                self._reorder_for_philosophical_flow
            ]
        
        # Пробуем разные стратегии
        original_line = line2
        for strategy in strategies:
            new_line = strategy(words2, line1, style)
            if new_line != original_line and grammar.is_grammatically_correct(new_line):
                return new_line
        
        return original_line
    
    def _add_lyrical_epithet(self, words: List[str], base_line: str, style: str) -> str:
        """Добавляет лирический эпитет"""
        if len(words) >= 2:
            last_word = words[-1]
            epithets = self.style_enhancements.get(style, {}).get("epithets", ["тихий", "нежный", "светлый"])
            
            if epithets:
                epithet = random.choice(epithets)
                
                # Согласуем эпитет с существительным
                agreed_epithet = grammar.make_adjective_agree(epithet, last_word)
                
                # Вставляем эпитет перед последним словом
                words.insert(-1, agreed_epithet)
        
        return " ".join(words)
    
    def _change_to_lyrical_word(self, words: List[str], base_line: str, style: str) -> str:
        """Меняет последнее слово на более лирическое"""
        if not words:
            return " ".join(words)
        
        lyrical_words = ["любовь", "душа", "мечта", "печаль", "надежда", "грусть", "радость"]
        new_word = random.choice(lyrical_words)
        
        # Склоняем новое слово к тому же падежу, что и старое
        old_word = words[-1]
        old_analysis = grammar.analyze_word(old_word)
        if old_analysis:
            target_case = old_analysis.get('case', 'nomn')
            target_number = old_analysis.get('number', 'sing')
            new_word = grammar.inflect_word(new_word, target_case, target_number)
        
        words[-1] = new_word
        return " ".join(words)
    
    def _add_adverb_to_line(self, words: List[str], base_line: str, style: str) -> str:
        """Добавляет наречие в строку"""
        if len(words) >= 2:
            adverbs = self.style_enhancements.get(style, {}).get("adverbs", ["тихо", "ясно", "светло"])
            
            if adverbs:
                adverb = random.choice(adverbs)
                
                # Вставляем наречие в случайное место (но не в конец)
                insert_pos = random.randint(0, max(0, len(words)-2))
                words.insert(insert_pos, adverb)
        
        return " ".join(words)
    
    def _reorder_for_lyrical_flow(self, words: List[str], base_line: str, style: str) -> str:
        """Переставляет слова для лучшего лирического потока"""
        if len(words) > 3:
            # Сохраняем первое и последнее слово
            first = words[0]
            last = words[-1]
            middle = words[1:-1]
            
            # Аккуратно переставляем слова в середине
            if len(middle) >= 2:
                # Меняем местами два случайных слова в середине
                if len(middle) >= 2:
                    i, j = random.sample(range(len(middle)), 2)
                    middle[i], middle[j] = middle[j], middle[i]
            
            words = [first] + middle + [last]
        
        return " ".join(words)
    
    def _add_humorous_element(self, words: List[str], base_line: str, style: str) -> str:
        """Добавляет юмористический элемент"""
        if len(words) >= 2:
            # Добавляем гиперболу
            hyperboles = self.style_enhancements.get(style, {}).get("hyperboles", ["до небес", "до упаду"])
            
            if hyperboles and random.random() > 0.5:
                hyperbole = random.choice(hyperboles)
                words[-1] = words[-1] + " " + hyperbole
            else:
                # Добавляем сравнение
                comparisons = self.style_enhancements.get(style, {}).get("comparisons", [])
                if comparisons:
                    comparison = random.choice(comparisons)
                    words.append(comparison)
        
        return " ".join(words)
    
    def _change_to_humorous_word(self, words: List[str], base_line: str, style: str) -> str:
        """Меняет слово на более юмористическое"""
        if not words:
            return " ".join(words)
        
        humorous_words = ["смех", "шутка", "анекдот", "прикол", "хохот", "курьез"]
        new_word = random.choice(humorous_words)
        
        # Сохраняем грамматическую форму
        old_word = words[-1]
        old_analysis = grammar.analyze_word(old_word)
        if old_analysis:
            target_case = old_analysis.get('case', 'nomn')
            target_number = old_analysis.get('number', 'sing')
            new_word = grammar.inflect_word(new_word, target_case, target_number)
        
        words[-1] = new_word
        return " ".join(words)
    
    def _add_exaggeration(self, words: List[str], base_line: str, style: str) -> str:
        """Добавляет преувеличение"""
        if len(words) >= 2:
            exaggerations = ["очень", "невероятно", "фантастически", "ужасно"]
            exaggeration = random.choice(exaggerations)
            
            # Вставляем перед последним словом
            words.insert(-1, exaggeration)
        
        return " ".join(words)
    
    def _reorder_for_humorous_effect(self, words: List[str], base_line: str, style: str) -> str:
        """Переставляет слова для юмористического эффекта"""
        if len(words) > 2:
            # Более активная перестановка для юмора
            random.shuffle(words)
        
        return " ".join(words)
    
    def _add_philosophical_depth(self, words: List[str], base_line: str, style: str) -> str:
        """Добавляет философскую глубину"""
        if len(words) >= 2:
            # Добавляем философский вопрос
            questions = self.style_enhancements.get(style, {}).get("questions", [" - вот в чем вопрос"])
            
            if questions and random.random() > 0.5:
                question = random.choice(questions)
                words[-1] = words[-1] + question
            else:
                # Добавляем абстракцию
                abstractions = self.style_enhancements.get(style, {}).get("abstractions", ["бытие", "сущность"])
                if abstractions:
                    abstraction = random.choice(abstractions)
                    
                    # Склоняем абстракцию
                    last_word = words[-1]
                    last_analysis = grammar.analyze_word(last_word)
                    if last_analysis:
                        target_case = last_analysis.get('case', 'nomn')
                        target_number = last_analysis.get('number', 'sing')
                        abstraction = grammar.inflect_word(abstraction, target_case, target_number)
                    
                    words.insert(-1, abstraction)
        
        return " ".join(words)
    
    def _change_to_philosophical_word(self, words: List[str], base_line: str, style: str) -> str:
        """Меняет слово на более философское"""
        if not words:
            return " ".join(words)
        
        philosophical_words = ["жизнь", "смерть", "время", "смысл", "истина", "вечность", "бытие"]
        new_word = random.choice(philosophical_words)
        
        # Сохраняем грамматическую форму
        old_word = words[-1]
        old_analysis = grammar.analyze_word(old_word)
        if old_analysis:
            target_case = old_analysis.get('case', 'nomn')
            target_number = old_analysis.get('number', 'sing')
            new_word = grammar.inflect_word(new_word, target_case, target_number)
        
        words[-1] = new_word
        return " ".join(words)
    
    def _add_profound_adverb(self, words: List[str], base_line: str, style: str) -> str:
        """Добавляет глубокомысленное наречие"""
        if len(words) >= 2:
            profound_adverbs = ["глубоко", "вечно", "сущностно", "абсолютно", "истинно"]
            adverb = random.choice(profound_adverbs)
            
            # Вставляем в начало строки
            words.insert(0, adverb)
        
        return " ".join(words)
    
    def _reorder_for_philosophical_flow(self, words: List[str], base_line: str, style: str) -> str:
        """Переставляет слова для философского звучания"""
        if len(words) > 3:
            # Для философского стиля делаем небольшие перестановки
            middle = words[1:-1]
            if len(middle) >= 2:
                # Меняем местами два соседних слова
                i = random.randint(0, len(middle)-2)
                middle[i], middle[i+1] = middle[i+1], middle[i]
            
            words = [words[0]] + middle + [words[-1]]
        
        return " ".join(words)
    
    def _improve_meter_with_grammar(self, lines: List[str], analyses: List[Dict]) -> Tuple[List[str], List[str]]:
        """Улучшает размер с учетом грамматики"""
        improvements = []
        
        if not lines:
            return lines, improvements
        
        # Определяем целевое количество слогов
        syllable_counts = [TextUtils.count_syllables(line) for line in lines]
        avg_syllables = sum(syllable_counts) / len(syllable_counts)
        target_syllables = round(avg_syllables)
        
        # Определяем основной размер
        meters = [TextUtils.detect_rhythm(line) for line in lines]
        main_meter = max(set(meters), key=meters.count) if meters else "ямб"
        
        for i, (line, analysis) in enumerate(zip(lines, analyses)):
            current_syllables = syllable_counts[i]
            current_meter = meters[i]
            
            needs_fix = False
            
            if current_meter != main_meter:
                needs_fix = True
            
            if abs(current_syllables - target_syllables) > 1:
                needs_fix = True
            
            if needs_fix:
                new_line = self._adjust_line_to_meter(line, analysis, main_meter, target_syllables)
                if new_line != line:
                    lines[i] = new_line
                    improvements.append(f"Строка {i+1}: скорректирован размер ({current_meter}->{main_meter})")
        
        return lines, improvements
    
    def _adjust_line_to_meter(self, line: str, analysis: Dict, 
                            target_meter: str, target_syllables: int) -> str:
        """Корректирует строку под заданный размер и количество слогов"""
        words = analysis["words"].copy()
        
        if not words:
            return line
        
        current_syllables = TextUtils.count_syllables(line)
        diff = target_syllables - current_syllables
        
        if diff > 0:
            # Нужно добавить слоги
            additions_by_meter = {
                "ямб": ["и", "вновь", "уже", "лишь", "ведь"],
                "хорей": ["тихий", "ясный", "светлый", "далекий"],
                "дактиль": ["тихонечко", "медленненько", "осторожненько"]
            }
            
            additions = additions_by_meter.get(target_meter, ["тихо", "ясно", "светло"])
            
            for _ in range(min(diff, 2)):
                addition = random.choice(additions)
                
                # Определяем грамматическую форму добавления
                if addition in ["и", "вновь", "уже", "лишь", "ведь"]:
                    # Это частицы/наречия, не требуют согласования
                    insert_pos = random.randint(0, len(words)-1)
                    words.insert(insert_pos, addition)
                else:
                    # Это прилагательные/наречия
                    if len(words) >= 2:
                        # Пробуем согласовать с соседним словом
                        insert_pos = random.randint(0, len(words)-1)
                        neighbor = words[insert_pos] if insert_pos < len(words) else words[-1]
                        neighbor_analysis = grammar.analyze_word(neighbor)
                        
                        if neighbor_analysis and neighbor_analysis.get('pos') == 'NOUN':
                            # Согласуем прилагательное с существительным
                            addition = grammar.make_adjective_agree(addition, neighbor)
                        
                        words.insert(insert_pos, addition)
        
        elif diff < 0:
            # Нужно убрать слоги
            if len(words) > 2 and abs(diff) >= 1:
                # Убираем короткие служебные слова
                removable = []
                for j, word in enumerate(words):
                    if j not in [0, len(words)-1] and len(word) <= 4:
                        word_analysis = grammar.analyze_word(word)
                        if word_analysis:
                            pos = word_analysis.get('pos')
                            if pos in ['CONJ', 'PRCL', 'ADVB']:  # Союзы, частицы, наречия
                                removable.append(j)
                
                if removable and abs(diff) >= 1:
                    remove_idx = random.choice(removable)
                    words.pop(remove_idx)
        
        # Корректируем для определенного размера
        if target_meter == "ямб":
            if len(words) % 2 != 0 and len(words) > 1:
                words.append("вновь")
        
        elif target_meter == "хорей":
            if len(words) < 3:
                additions = ["тихо", "ясно", "светло"]
                words.append(random.choice(additions))
        
        return " ".join(words)
    
    def _improve_style_with_natasha(self, lines: List[str], target_style: str, 
                                  prompt: str) -> Tuple[List[str], List[str]]:
        """Улучшает стиль с использованием Natasha"""
        improvements = []
        
        if target_style not in self.style_enhancements:
            return lines, improvements
        
        style_data = self.style_enhancements[target_style]
        style_words = self.styles.get(target_style, {}).get("keywords", [])
        
        # Проверяем, есть ли ключевые слова стиля в стихе
        poem_text = " ".join(lines).lower()
        existing_style_words = [w for w in style_words if w in poem_text]
        
        # Добавляем недостающие ключевые слова
        words_to_add = [w for w in style_words if w not in existing_style_words]
        
        if words_to_add:
            # Ограничиваем количество добавляемых слов
            words_to_add = random.sample(words_to_add, min(2, len(words_to_add)))
            
            for word in words_to_add:
                # Ищем подходящую строку для добавления слова
                suitable_lines = []
                for i, line in enumerate(lines):
                    line_words = line.split()
                    if len(line_words) > 2 and word not in line.lower():
                        suitable_lines.append(i)
                
                if suitable_lines:
                    line_idx = random.choice(suitable_lines)
                    words = lines[line_idx].split()
                    
                    # Определяем грамматическую форму слова
                    word_analysis = grammar.analyze_word(word)
                    if word_analysis:
                        # Пробуем определить подходящий падеж
                        if len(words) >= 2:
                            neighbor = words[-1]  # Берем последнее слово как ориентир
                            neighbor_analysis = grammar.analyze_word(neighbor)
                            
                            if neighbor_analysis:
                                target_case = neighbor_analysis.get('case', 'nomn')
                                target_number = neighbor_analysis.get('number', 'sing')
                                word = grammar.inflect_word(word, target_case, target_number)
                    
                    # Вставляем слово
                    insert_pos = random.randint(0, len(words)-1)
                    words.insert(insert_pos, word)
                    lines[line_idx] = " ".join(words)
                    
                    improvements.append(f"Строка {line_idx+1}: добавлено слово стиля '{word}'")
        
        # Добавляем стилевые элементы (эпитеты, сравнения и т.д.)
        for i, line in enumerate(lines):
            if random.random() > 0.7:  # 30% вероятность улучшения строки
                enhanced_line = self._add_style_element(line, target_style)
                if enhanced_line != line:
                    lines[i] = enhanced_line
                    improvements.append(f"Строка {i+1}: добавлен стилевой элемент")
        
        return lines, improvements
    
    def _add_style_element(self, line: str, style: str) -> str:
        """Добавляет стилевой элемент к строке"""
        if style not in self.style_enhancements:
            return line
        
        style_data = self.style_enhancements[style]
        words = line.split()
        
        if len(words) < 2:
            return line
        
        # Выбираем тип стилевого элемента
        element_types = []
        
        if style_data.get("epithets"):
            element_types.append("epithet")
        
        if style_data.get("comparisons") and len(words) < 6:
            element_types.append("comparison")
        
        if not element_types:
            return line
        
        element_type = random.choice(element_types)
        
        if element_type == "epithet":
            # Добавляем эпитет
            epithets = style_data["epithets"]
            epithet = random.choice(epithets)
            
            # Согласуем с последним существительным в строке
            for j in range(len(words)-1, -1, -1):
                word_analysis = grammar.analyze_word(words[j])
                if word_analysis and word_analysis.get('pos') == 'NOUN':
                    epithet = grammar.make_adjective_agree(epithet, words[j])
                    words.insert(j, epithet)
                    break
        
        elif element_type == "comparison":
            # Добавляем сравнение
            comparisons = style_data["comparisons"]
            comparison = random.choice(comparisons)
            
            # Добавляем в конец строки
            if line[-1] in ",.!?;:":
                line = line[:-1]
            
            line = line + ", " + comparison
        
        return " ".join(words) if element_type == "epithet" else line
    
    def _improve_relevance_with_natasha(self, lines: List[str], prompt: str) -> Tuple[List[str], List[str]]:
        """Улучшает соответствие запросу с использованием Natasha"""
        improvements = []
        
        if not prompt:
            return lines, improvements
        
        # Извлекаем ключевые слова из запроса
        prompt_words = re.findall(r'[а-яА-ЯёЁ]{3,}', prompt.lower())
        
        if not prompt_words:
            return lines, improvements
        
        poem_text = " ".join(lines).lower()
        existing_words = [w for w in prompt_words if w in poem_text]
        words_to_add = [w for w in prompt_words if w not in existing_words]
        
        if words_to_add:
            # Ограничиваем количество добавляемых слов
            words_to_add = sorted(words_to_add, key=len, reverse=True)[:2]
            
            for word in words_to_add:
                # Ищем подходящую строку
                suitable_lines = []
                for i, line in enumerate(lines):
                    line_words = line.split()
                    if len(line_words) > 2 and word not in line.lower():
                        suitable_lines.append(i)
                
                if suitable_lines:
                    line_idx = random.choice(suitable_lines)
                    words = lines[line_idx].split()
                    
                    # Анализируем контекст для определения грамматической формы
                    if len(words) >= 2:
                        # Пробуем определить падеж по соседним словам
                        context_word = words[-1] if random.random() > 0.5 else words[0]
                        context_analysis = grammar.analyze_word(context_word)
                        
                        if context_analysis:
                            target_case = context_analysis.get('case', 'nomn')
                            target_number = context_analysis.get('number', 'sing')
                            word = grammar.inflect_word(word, target_case, target_number)
                    
                    # Вставляем слово
                    if random.random() > 0.7:
                        words.insert(0, word.capitalize() if line_idx == 0 else word)
                    elif random.random() > 0.5:
                        insert_pos = random.randint(1, len(words)-1)
                        words.insert(insert_pos, word)
                    else:
                        words.insert(-1, word)
                    
                    lines[line_idx] = " ".join(words)
                    improvements.append(f"Строка {line_idx+1}: добавлено слово из запроса '{word}'")
        
        return lines, improvements
    
    def _fix_grammar_errors(self, lines: List[str], analyses: List[Dict]) -> Tuple[List[str], List[str]]:
        """Исправляет грамматические ошибки"""
        improvements = []
        
        for i, analysis in enumerate(analyses):
            line = analysis["text"]
            words = analysis["words"]
            word_analyses = analysis["analyses"]
            
            if len(words) < 2:
                continue
            
            # Проверяем согласование внутри строки
            for j in range(1, len(words)):
                if j < len(word_analyses):
                    analysis1 = word_analyses[j-1]
                    analysis2 = word_analyses[j]
                    
                    if analysis1 and analysis2:
                        # Исправляем согласование прилагательного и существительного
                        if (analysis1.get('pos') == 'ADJ' and analysis2.get('pos') == 'NOUN' and
                            not self._check_adjective_noun_agreement(analysis1, analysis2)):
                            
                            # Исправляем прилагательное
                            corrected_adj = grammar.make_adjective_agree(
                                words[j-1], words[j]
                            )
                            
                            if corrected_adj != words[j-1]:
                                words[j-1] = corrected_adj
                                improvements.append(
                                    f"Строка {i+1}: исправлено согласование '{words[j-1]}' с '{words[j]}'"
                                )
            
            # Если были изменения, обновляем строку
            new_line = " ".join(words)
            if new_line != line:
                lines[i] = new_line
        
        return lines, improvements
    
    def _enhance_line_with_grammar(self, analysis: Dict, style: str) -> str:
        """Усиливает строку с учетом грамматики и стиля"""
        line = analysis["text"]
        words = analysis["words"].copy()
        
        if len(words) < 2:
            return line
        
        # Выбираем стратегию улучшения в зависимости от стиля
        if style == "лирический":
            enhancements = [
                self._add_lyrical_touch,
                self._improve_verbs_lyrical,
                self._add_poetic_imagery
            ]
        elif style == "юмористический":
            enhancements = [
                self._add_humorous_touch,
                self._improve_verbs_humorous,
                self._add_comic_element
            ]
        else:  # философский
            enhancements = [
                self._add_philosophical_touch,
                self._improve_verbs_philosophical,
                self._add_profound_element
            ]
        
        # Пробуем разные улучшения
        original_line = line
        for enhancement in enhancements:
            enhanced_words = enhancement(words, style)
            if enhanced_words != words:
                words = enhanced_words
                new_line = " ".join(words)
                if grammar.is_grammatically_correct(new_line):
                    return new_line
        
        return original_line
    
    def _add_lyrical_touch(self, words: List[str], style: str) -> List[str]:
        """Добавляет лирический оттенок"""
        if len(words) >= 2:
            # Добавляем нежное наречие
            gentle_adverbs = ["тихо", "нежно", "грустно", "светло", "печально"]
            adverb = random.choice(gentle_adverbs)
            
            insert_pos = random.randint(0, len(words)-1)
            words.insert(insert_pos, adverb)
        
        return words
    
    def _improve_verbs_lyrical(self, words: List[str], style: str) -> List[str]:
        """Улучшает глаголы для лирического стиля"""
        lyrical_verbs = self.verbs_by_style.get("лирический", [])
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.common_synonyms:
                # Заменяем обычный глагол на более лирический
                if lyrical_verbs:
                    new_verb = random.choice(lyrical_verbs)
                    
                    # Склоняем глагол
                    word_analysis = grammar.analyze_word(word)
                    if word_analysis:
                        # Для глаголов сохраняем время и лицо (упрощенно)
                        words[i] = new_verb
        
        return words
    
    def _add_poetic_imagery(self, words: List[str], style: str) -> List[str]:
        """Добавляет поэтический образ"""
        if len(words) >= 3:
            imagery = ["как сон", "как тень", "как эхо", "как грёза"]
            image = random.choice(imagery)
            words.append(image)
        
        return words
    
    def _add_humorous_touch(self, words: List[str], style: str) -> List[str]:
        """Добавляет юмористический оттенок"""
        if len(words) >= 2:
            # Добавляем смешное слово
            funny_words = ["смешно", "забавно", "странно", "курьезно"]
            funny_word = random.choice(funny_words)
            
            insert_pos = random.randint(0, len(words)-1)
            words.insert(insert_pos, funny_word)
        
        return words
    
    def _improve_verbs_humorous(self, words: List[str], style: str) -> List[str]:
        """Улучшает глаголы для юмористического стиля"""
        humorous_verbs = self.verbs_by_style.get("юмористический", [])
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.common_synonyms:
                # Заменяем обычный глагол на более юмористический
                if humorous_verbs:
                    new_verb = random.choice(humorous_verbs)
                    words[i] = new_verb
        
        return words
    
    def _add_comic_element(self, words: List[str], style: str) -> List[str]:
        """Добавляет комический элемент"""
        if len(words) >= 3:
            comic_elements = ["вверх ногами", "как попало", "не к месту"]
            element = random.choice(comic_elements)
            words.append(element)
        
        return words
    
    def _add_philosophical_touch(self, words: List[str], style: str) -> List[str]:
        """Добавляет философский оттенок"""
        if len(words) >= 2:
            # Добавляем глубокомысленное слово
            profound_words = ["глубоко", "вечно", "сущностно", "абсолютно"]
            profound_word = random.choice(profound_words)
            
            insert_pos = 0  # В начало для философского звучания
            words.insert(insert_pos, profound_word)
        
        return words
    
    def _improve_verbs_philosophical(self, words: List[str], style: str) -> List[str]:
        """Улучшает глаголы для философского стиля"""
        philosophical_verbs = self.verbs_by_style.get("философский", [])
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            if word_lower in self.common_synonyms:
                # Заменяем обычный глагол на более философский
                if philosophical_verbs:
                    new_verb = random.choice(philosophical_verbs)
                    words[i] = new_verb
        
        return words
    
    def _add_profound_element(self, words: List[str], style: str) -> List[str]:
        """Добавляет глубокомысленный элемент"""
        if len(words) >= 3:
            profound_elements = ["в сущности", "по сути", "в конечном счете"]
            element = random.choice(profound_elements)
            words.append(element)
        
        return words
    
    def _final_rhyme_check(self, lines: List[str], style: str) -> Tuple[List[str], List[str]]:
        """Финальная проверка и улучшение рифм"""
        improvements = []
        
        if len(lines) < 2:
            return lines, improvements
        
        # Проверяем качество рифм в конце работы
        for i in range(len(lines)-1):
            word1 = TextUtils.extract_last_word_grammatical(lines[i])
            word2 = TextUtils.extract_last_word_grammatical(lines[i+1])
            
            if word1 and word2:
                if not grammar.words_rhyme(word1, word2, strict=True):
                    # Пробуем улучшить рифму
                    analysis1 = grammar.analyze_word(word1)
                    if analysis1:
                        rhyming_words = self._find_grammatical_rhymes(word1, analysis1, style)
                        if rhyming_words:
                            best_rhyme = self._select_best_grammatical_rhyme(word1, analysis1, rhyming_words, lines[i+1])
                            if best_rhyme:
                                # Приводим к нужной форме
                                target_case = analysis1.get('case', 'nomn')
                                target_number = analysis1.get('number', 'sing')
                                inflected_rhyme = grammar.inflect_word(best_rhyme, target_case, target_number)
                                
                                # Заменяем последнее слово
                                words = lines[i+1].split()
                                if words:
                                    words[-1] = inflected_rhyme
                                    lines[i+1] = " ".join(words)
                                    improvements.append(f"Улучшена рифма в строках {i+1}-{i+2}")
        
        return lines, improvements
    
    def _ensure_grammatical_consistency(self, lines: List[str]) -> List[str]:
        """Обеспечивает грамматическую целостность стиха"""
        cleaned_lines = []
        
        for line in lines:
            # Убираем лишние знаки препинания в начале
            while line and line[0] in ",.!?;:— ":
                line = line[1:]
            
            # Убираем лишние знаки препинания в конце
            while len(line) > 1 and line[-1] in ",.!?;:— " and line[-2] in ",.!?;:— ":
                line = line[:-1]
            
            # Капитализируем первую букву, если нужно
            if line and line[0].islower():
                line = line[0].upper() + line[1:]
            
            cleaned_lines.append(line)
        
        # Последняя строка должна заканчиваться точкой или многоточием
        if cleaned_lines:
            last_line = cleaned_lines[-1]
            if last_line and last_line[-1] not in ".!?…":
                cleaned_lines[-1] = last_line + ("..." if random.random() > 0.5 else ".")
        
        return cleaned_lines
    
    def _improve_punctuation(self, lines: List[str]) -> List[str]:
        """Улучшает пунктуацию"""
        for i in range(len(lines)):
            line = lines[i].strip()
            
            if not line:
                continue
            
            # Убираем лишние знаки препинания
            while line and line[-1] in ",.!?;:—":
                line = line[:-1]
            
            # Добавляем знак препинания в зависимости от позиции
            if i < len(lines) - 1:
                # Не последняя строка
                punctuation_options = [",", ",", ";", " —", ":"]
                if random.random() > 0.4:  # 60% вероятность
                    line = line + random.choice(punctuation_options)
            else:
                # Последняя строка
                line = line + ("..." if random.random() > 0.5 else ".")
            
            lines[i] = line
        
        return lines