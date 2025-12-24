"""
Модуль для работы с русской грамматикой с использованием Natasha
"""

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)
from typing import List, Dict, Tuple, Optional
import re

class NatashaGrammar:
    """Класс для работы с русской грамматикой с использованием Natasha"""
    
    def __init__(self):
        # Инициализируем компоненты Natasha с актуальным API
        try:
            self.segmenter = Segmenter()
            self.morph_vocab = MorphVocab()
            emb = NewsEmbedding()
            self.morph_tagger = NewsMorphTagger(emb)
            print("✅ Natasha инициализирована")
        except Exception as e:
            print(f"⚠️ Ошибка инициализации Natasha: {e}")
            self.segmenter = None
            self.morph_vocab = None
            self.morph_tagger = None
        
        # Кэш для ускорения работы
        self._cache = {}
    
    def analyze_word(self, word: str) -> Optional[Dict]:
        """Анализирует слово и возвращает его грамматические характеристики"""
        if not word or not word.strip():
            return None
        
        # Проверяем кэш
        cache_key = word.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key].copy()
        
        # Если Natasha не инициализирована, возвращаем простой анализ
        if not self.morph_tagger or not self.segmenter:
            return self._simple_analysis(word)
        
        try:
            # Создаем документ Natasha с текущими API
            doc = Doc(word)
            doc.segment(self.segmenter)  # Используем сегментатор
            doc.tag_morph(self.morph_tagger)
            
            if not doc.tokens:
                return self._simple_analysis(word)
            
            token = doc.tokens[0]
            
            # Извлекаем информацию
            result = {
                'text': word,
                'normal': word.lower(),
                'pos': 'X',
                'feats': {},  # Грамматические признаки
                'gender': None,
                'number': None,
                'case': None
            }
            
            # Если есть морфологический разбор
            if hasattr(token, 'pos') and token.pos:
                result['pos'] = token.pos
                
                # Получаем лемму через morph_vocab
                if hasattr(token, 'lemma') and token.lemma:
                    result['normal'] = token.lemma.lower()
                else:
                    # Используем pymorphy2 как fallback
                    try:
                        import pymorphy2
                        morph = pymorphy2.MorphAnalyzer()
                        parsed = morph.parse(word)[0]
                        result['normal'] = parsed.normal_form
                    except:
                        pass
                
                # Извлекаем грамматические признаки
                if hasattr(token, 'feats') and token.feats:
                    feats_dict = {}
                    
                    # Обработка признаков в зависимости от их формата
                    if isinstance(token.feats, dict):
                        feats_dict = token.feats
                    elif hasattr(token.feats, 'items'):
                        feats_dict = dict(token.feats.items())
                    elif isinstance(token.feats, str):
                        for feat in token.feats.split('|'):
                            if '=' in feat:
                                key, value = feat.split('=', 1)
                                feats_dict[key] = value
                    
                    result['feats'] = feats_dict
                    result['gender'] = feats_dict.get('Gender')
                    result['number'] = feats_dict.get('Number')
                    result['case'] = feats_dict.get('Case')
            
            # Сохраняем в кэш
            self._cache[cache_key] = result.copy()
            return result
            
        except Exception as e:
            print(f"⚠️ Ошибка анализа слова '{word}': {e}")
            return self._simple_analysis(word)
    
    def _simple_analysis(self, word: str) -> Dict:
        """Простой анализ слова без Natasha"""
        word_lower = word.lower().strip()
        
        # Пропускаем пунктуацию и короткие слова
        if len(word_lower) <= 1 or word_lower in ['.', ',', '!', '?', ':', ';', '-', '—', '...']:
            return {
                'text': word,
                'normal': word_lower,
                'pos': 'PUNCT',
                'gender': None,
                'number': None,
                'case': None,
                'feats': {}
            }
        
        # Определяем часть речи по окончанию
        pos = 'NOUN'
        if word_lower.endswith(('ый', 'ий', 'ой', 'ая', 'яя', 'ое', 'ее', 'ые', 'ие')):
            pos = 'ADJ'
        elif word_lower.endswith(('ть', 'ться', 'тись', 'чь', 'л', 'ла', 'ло', 'ли', 'ем', 'ет', 'ют', 'им', 'ит', 'ат')):
            pos = 'VERB'
        elif word_lower.endswith(('о', 'е', 'ы', 'и')):
            pos = 'ADV'
        elif word_lower in ['и', 'или', 'а', 'но', 'да', 'либо', 'нибудь', 'же', 'ли', 'бы', 'б']:
            pos = 'CONJ'
        elif word_lower in ['в', 'на', 'под', 'над', 'с', 'со', 'из', 'от', 'до', 'по', 'к', 'ко', 'у', 'о', 'об', 'обо']:
            pos = 'PREP'
        
        # Определяем род по окончанию
        gender = 'masc'
        if word_lower.endswith(('а', 'я', 'ь')) and pos in ['NOUN', 'ADJ']:
            if word_lower.endswith(('а', 'я')):
                gender = 'femn'
            elif word_lower.endswith('ь'):
                # Проверяем типичные окончания для женского рода
                feminine_endings = ['ость', 'бь', 'вь', 'дь', 'зь', 'ль', 'мь', 'нь', 'пь', 'рь', 'сь', 'ть', 'фь', 'шь', 'щь']
                if any(word_lower.endswith(ending) for ending in feminine_endings):
                    gender = 'femn'
                else:
                    gender = 'masc'
        elif word_lower.endswith(('о', 'е', 'мя')):
            gender = 'neut'
        
        # Определяем число
        number = 'sing'
        if word_lower.endswith(('ы', 'и', 'а', 'я')) and pos in ['NOUN', 'ADJ']:
            # Исключения для единственного числа
            singular_exceptions = ['папа', 'мама', 'дядя', 'тетя', 'время', 'имя', 'племя', 'семя', 'стремя']
            if word_lower not in singular_exceptions:
                number = 'plur'
        
        return {
            'text': word,
            'normal': word_lower,
            'pos': pos,
            'gender': gender,
            'number': number,
            'case': 'nomn',
            'feats': {}
        }
    
    def get_normal_form(self, word: str) -> str:
        """Получает нормальную форму слова (лемму)"""
        analysis = self.analyze_word(word)
        return analysis['normal'] if analysis else word.lower()
    
    def get_pos(self, word: str) -> str:
        """Получает часть речи слова"""
        analysis = self.analyze_word(word)
        return analysis['pos'] if analysis else 'X'
    
    def get_case(self, word: str) -> Optional[str]:
        """Получает падеж слова"""
        analysis = self.analyze_word(word)
        return analysis['case'] if analysis else None
    
    def get_number(self, word: str) -> Optional[str]:
        """Получает число слова"""
        analysis = self.analyze_word(word)
        return analysis['number'] if analysis else None
    
    def get_gender(self, word: str) -> Optional[str]:
        """Получает род слова"""
        analysis = self.analyze_word(word)
        return analysis['gender'] if analysis else None
    
    def inflect_word(self, word: str, target_case: str = None, 
                    target_number: str = None) -> str:
        """Склоняет слово к нужному падежу и числу (упрощенно)"""
        # Для упрощения возвращаем исходное слово
        # В реальной реализации нужно использовать pymorphy2 или аналоги
        return word
    
    def words_rhyme(self, word1: str, word2: str, strict: bool = False) -> bool:
        """Проверяет, рифмуются ли два слова"""
        if not word1 or not word2:
            return False
        
        w1 = word1.lower().strip()
        w2 = word2.lower().strip()
        
        if len(w1) < 2 or len(w2) < 2:
            return False
        
        # Удаляем знаки препинания
        w1 = re.sub(r'[.,!?;:—\-]+$', '', w1)
        w2 = re.sub(r'[.,!?;:—\-]+$', '', w2)
        
        # Проверяем фонетическую рифму (по окончаниям)
        vowels = 'аеёиоуыэюя'
        
        # Извлекаем гласные звуки
        def extract_vowel_pattern(w):
            pattern = []
            for char in w:
                if char in vowels:
                    pattern.append(char)
            return pattern
        
        v1 = extract_vowel_pattern(w1)
        v2 = extract_vowel_pattern(w2)
        
        # Если паттерны гласных разные, не рифмуются
        if v1 != v2:
            return False
        
        # Проверяем совпадение последних 2-3 букв
        for length in [3, 2]:
            if len(w1) >= length and len(w2) >= length:
                if w1[-length:] == w2[-length:]:
                    return True
        
        # Для нестрогой проверки: проверяем созвучие окончаний
        if not strict:
            min_len = min(len(w1), len(w2))
            if min_len >= 3:
                # Сравниваем последние 2-3 символа
                if w1[-2:] == w2[-2:] or w1[-3:] == w2[-3:]:
                    return True
        
        return False
    
    def find_rhyming_words(self, word: str, word_list: List[str], 
                          max_results: int = 10) -> List[str]:
        """Находит рифмующиеся слова в списке"""
        if not word or not word_list:
            return []
        
        rhymes = []
        for w in word_list:
            if w != word and self.words_rhyme(word, w):
                rhymes.append(w)
                if len(rhymes) >= max_results:
                    break
        
        return rhymes
    
    def make_adjective_agree(self, adjective: str, noun: str) -> str:
        """Согласует прилагательное с существительным (упрощенно)"""
        # Упрощенная версия - возвращаем прилагательное как есть
        # В реальной реализации нужно использовать pymorphy2
        return adjective
    
    def is_grammatically_correct(self, phrase: str) -> bool:
        """Проверяет грамматическую корректность фразы"""
        try:
            # Простая проверка
            words = phrase.split()
            if len(words) < 1:
                return False
            
            # Проверяем, что нет очевидных ошибок
            if '  ' in phrase:  # Двойные пробелы
                return False
            
            return True
        except:
            return False
    
    def get_word_stress_pattern(self, word: str) -> List[int]:
        """Возвращает паттерн ударений слова"""
        vowels = 'аеёиоуыэюя'
        pattern = []
        
        for i, char in enumerate(word.lower()):
            if char in vowels:
                pattern.append(i)
        
        return pattern

# Создаем глобальный экземпляр
grammar = NatashaGrammar()