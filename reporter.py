"""
Модуль генерации отчетов об улучшении стихов
"""

import datetime
from typing import Dict, List, Tuple
from utils import TextUtils

class ReportGenerator:
    """Генератор отчетов об улучшении"""
    
    def __init__(self):
        self.report_count = 0
    
    def generate_report(self, original_poem: str, improved_poem: str,
                       original_analysis: Dict, improved_analysis: Dict,
                       improvements: List[str], prompt: str, style: str) -> str:
        """Создает полный отчет об улучшении стиха"""
        
        self.report_count += 1
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_id = f"RPT-{self.report_count:04d}"
        
        report = f"""
{'='*80}
ОТЧЕТ ОБ УЛУЧШЕНИИ ПОЭТИЧЕСКОГО АГЕНТА
{'='*80}

Идентификатор: {report_id}
Время генерации: {timestamp}
Запрос: "{prompt}"
Стиль: {style}

1. СРАВНЕНИЕ СТИХОВ:
{'='*80}

ИСХОДНЫЙ СТИХ (Оценка: {original_analysis.get('score', 0):.2f}/1.0):
{'-'*40}
{self._format_poem_for_report(original_poem)}

УЛУЧШЕННЫЙ СТИХ (Оценка: {improved_analysis.get('score', 0):.2f}/1.0):
{'-'*40}
{self._format_poem_for_report(improved_poem)}

2. ДИАГНОСТИКА И УЛУЧШЕНИЯ:
{'='*80}

Найденные проблемы в исходном стихе:
{self._format_list(original_analysis.get('diagnostics', ['Проблем не обнаружено']))}

Выполненные улучшения:
{self._format_list(improvements if improvements else ['Улучшения не требовались'])}

3. ПОДРОБНЫЙ АНАЛИЗ КРИТЕРИЕВ:
{'='*80}
{self._create_detailed_comparison(original_analysis, improved_analysis)}

4. КОЛИЧЕСТВЕННЫЕ ПОКАЗАТЕЛИ:
{'='*80}
{self._create_metrics_table(original_analysis, improved_analysis)}

5. РЕКОМЕНДАЦИИ И ВЫВОДЫ:
{'='*80}
{self._generate_recommendations(original_analysis, improved_analysis, improvements)}

6. ПРИМЕРЫ И АНТИПРИМЕРЫ:
{'='*80}
{self._generate_examples_and_counterexamples(prompt, style)}

{'='*80}
Отчет сгенерирован автоматически поэтическим агентом.
{'='*80}
"""
        return report
    
    def generate_short_report(self, original_poem: str, improved_poem: str,
                            original_analysis: Dict, improved_analysis: Dict,
                            improvements: List[str]) -> str:
        """Создает краткий отчет"""
        score_diff = improved_analysis.get('score', 0) - original_analysis.get('score', 0)
        
        report = f"""📊 КРАТКИЙ ОТЧЕТ

ОЦЕНКА: {original_analysis.get('score', 0):.2f} → {improved_analysis.get('score', 0):.2f} 
         ({'+' if score_diff > 0 else ''}{score_diff:.2f})

УЛУЧШЕНИЯ: {len(improvements) if improvements else 0}

ОСНОВНЫЕ ИЗМЕНЕНИЯ:
{self._format_short_improvements(improvements[:3])}

РЕЗУЛЬТАТ: {self._get_improvement_summary(score_diff)}"""
        
        return report
    
    def _format_poem_for_report(self, poem: str) -> str:
        """Форматирует стих для отчета"""
        lines = poem.split('\n')
        formatted = ""
        for i, line in enumerate(lines, 1):
            formatted += f"{i:2d}. {line}\n"
        return formatted.rstrip()
    
    def _format_list(self, items: List[str]) -> str:
        """Форматирует список"""
        if not items:
            return "  - Нет данных\n"
        
        formatted = ""
        for i, item in enumerate(items, 1):
            prefix = "✓ " if "не требуются" in item.lower() else "• "
            formatted += f"  {prefix}{item}\n"
        return formatted
    
    def _format_short_improvements(self, improvements: List[str]) -> str:
        """Форматирует краткий список улучшений"""
        if not improvements:
            return "  Нет улучшений"
        
        formatted = ""
        for imp in improvements[:3]:
            if len(imp) > 60:
                imp = imp[:57] + "..."
            formatted += f"  • {imp}\n"
        return formatted
    
    def _create_detailed_comparison(self, orig: Dict, impr: Dict) -> str:
        """Создает детальное сравнение по критериям"""
        
        def format_criterion(name: str, orig_data: Dict, impr_data: Dict) -> str:
            orig_ok = orig_data.get('is_ok', False)
            impr_ok = impr_data.get('is_ok', False)
            
            result = f"{name.upper()}:\n"
            result += f"  До: {'✅ ПРОЙДЕНО' if orig_ok else '❌ НЕ ПРОЙДЕНО'}\n"
            result += f"  После: {'✅ ПРОЙДЕНО' if impr_ok else '❌ НЕ ПРОЙДЕНО'}\n"
            
            if name == "рифма":
                result += f"  Тип рифмы: {orig_data.get('rhyme_type', 'нет')} → {impr_data.get('rhyme_type', 'нет')}\n"
            
            elif name == "размер":
                result += f"  Размер: {orig_data.get('meter', 'не опр.')} → {impr_data.get('meter', 'не опр.')}\n"
                result += f"  Слоги: {orig_data.get('avg_syllables', 0)} → {impr_data.get('avg_syllables', 0)} в среднем\n"
            
            elif name == "стиль":
                orig_match = orig_data.get('style_match', 0)
                impr_match = impr_data.get('style_match', 0)
                result += f"  Соответствие: {orig_match:.2f} → {impr_match:.2f}\n"
                if 'found_keywords' in impr_data:
                    keywords = ', '.join(impr_data['found_keywords'][:3])
                    if len(impr_data['found_keywords']) > 3:  # ИСПРАВЛЕНО
                        keywords += "..."
                    result += f"  Ключевые слова: {keywords}\n"
            
            elif name == "цель":
                orig_sim = orig_data.get('similarity', 0)
                impr_sim = impr_data.get('similarity', 0)
                result += f"  Схожесть: {orig_sim:.2f} → {impr_sim:.2f}\n"
                if 'common_words' in impr_data:
                    words = ', '.join(impr_data['common_words'][:3])
                    if len(impr_data['common_words']) > 3:  # ИСПРАВЛЕНО
                        words += "..."
                    result += f"  Общие слова: {words}\n"
            
            return result + "\n"
        
        comparison = ""
        criteria = [
            ("рифма", orig.get('rhyme', {}), impr.get('rhyme', {})),
            ("размер", orig.get('meter', {}), impr.get('meter', {})),
            ("стиль", orig.get('style', {}), impr.get('style', {})),
            ("цель", orig.get('goal', {}), impr.get('goal', {}))
        ]
        
        for name, orig_data, impr_data in criteria:
            comparison += format_criterion(name, orig_data, impr_data)
        
        return comparison
    
    def _create_metrics_table(self, orig: Dict, impr: Dict) -> str:
        """Создает таблицу метрик"""
        
        metrics = [
            ("Общая оценка", orig.get('score', 0), impr.get('score', 0), "1.0"),
            ("Рифма", 1 if orig.get('rhyme', {}).get('is_ok') else 0, 
                    1 if impr.get('rhyme', {}).get('is_ok') else 0, "бинарная"),
            ("Размер", 1 if orig.get('meter', {}).get('is_ok') else 0,
                    1 if impr.get('meter', {}).get('is_ok') else 0, "бинарная"),
            ("Соответствие стилю", orig.get('style', {}).get('style_match', 0),
                    impr.get('style', {}).get('style_match', 0), "0-1"),
            ("Соответствие цели", orig.get('goal', {}).get('similarity', 0),
                    impr.get('goal', {}).get('similarity', 0), "0-1"),
            ("Количество строк", orig.get('line_count', 0),
                    impr.get('line_count', 0), "шт.")
        ]
        
        table = "Метрика                | До     | После  | Δ      | Шкала\n"
        table += "-" * 60 + "\n"
        
        for name, before, after, scale in metrics:
            delta = after - before
            delta_str = f"{delta:+.2f}" if isinstance(delta, float) else f"{delta:+d}"
            
            if isinstance(before, float):
                before_str = f"{before:.2f}"
                after_str = f"{after:.2f}"
            else:
                before_str = str(before)
                after_str = str(after)
            
            table += f"{name:<22} | {before_str:>6} | {after_str:>6} | {delta_str:>6} | {scale}\n"
        
        total_improvement = impr.get('score', 0) - orig.get('score', 0)
        improvement_percent = (total_improvement / max(orig.get('score', 0.01), 0.01)) * 100
        
        table += "\n" + "-" * 60 + "\n"
        table += f"ИТОГО: Улучшение на {total_improvement:.2f} баллов ({improvement_percent:.1f}%)\n"
        
        return table
    
    def _generate_recommendations(self, orig: Dict, impr: Dict, improvements: List[str]) -> str:
        """Генерирует рекомендации"""
        recommendations = []
        final_score = impr.get('score', 0)
        
        if final_score < 0.5:
            recommendations.append("● Рекомендуется полная переработка стиха")
            recommendations.append("● Увеличить количество строк до 8-12")
            recommendations.append("● Использовать более выразительные образы")
        
        elif final_score < 0.7:
            recommendations.append("● Можно улучшить рифмовку (использовать ABAB вместо AABB)")
            recommendations.append("● Добавить больше метафор и сравнений")
            recommendations.append("● Проверить единообразие размера во всех строках")
        
        elif final_score < 0.9:
            recommendations.append("● Экспериментировать с более сложными рифмами")
            recommendations.append("● Добавить внутренние рифмы")
            recommendations.append("● Использовать аллитерацию (повторение звуков)")
        
        else:
            recommendations.append("● Стих хорошего качества, можно считать завершенным")
            recommendations.append("● Для дальнейшего улучшения: поэкспериментировать с размером")
            recommendations.append("● Можно попробовать другие стили для сравнения")
        
        if improvements:
            rhyme_improvements = [i for i in improvements if "рифм" in i.lower()]
            meter_improvements = [i for i in improvements if "размер" in i.lower()]
            style_improvements = [i for i in improvements if "стиль" in i.lower()]
            
            if rhyme_improvements and len(rhyme_improvements) > 1:
                recommendations.append("● Обратить внимание на систему рифмовки")
            
            if meter_improvements:
                recommendations.append("● Поработать над ритмической структурой")
        
        style_data = impr.get('style', {})
        style_match = style_data.get('style_match', 0)
        if style_match < 0.6:
            recommendations.append("● Усилить соответствие выбранному стилю")
        
        goal_data = impr.get('goal', {})
        similarity = goal_data.get('similarity', 0)
        if similarity < 0.4:
            recommendations.append("● Усилить связь с исходным запросом")
        
        formatted_recs = "\n".join(recommendations)
        
        return f"""РЕКОМЕНДАЦИИ ДЛЯ ДАЛЬНЕЙШЕГО УЛУЧШЕНИЯ:

{formatted_recs}

ОБЩАЯ ОЦЕНКА КАЧЕСТВА: {self._get_quality_label(final_score)}"""
    
    def _generate_examples_and_counterexamples(self, prompt: str, style: str) -> str:
        """Генерирует примеры и антипримеры"""
        
        examples_by_style = {
            "лирический": {
                "хорошие": [
                    "Луна сияет над рекой, // И ветер шепчет нам с тобой.",
                    "Любовь в душе горит огнем, // И счастье длится целый день.",
                    "Печаль тихонько в сердце льется, // И ночь над миром опускается."
                ],
                "плохие": [
                    "Луна светит на небе. // Это красиво видеть.",
                    "Я люблю тебя очень. // Ты мне очень нравишься.",
                    "Грустно стало вечером. // Сижу один в комнате."
                ]
            },
            "юмористический": {
                "хорошие": [
                    "Кот в штанах гулял по крыше, // Мышь смеялась тихо в норке.",
                    "Слон танцует на матрасе, // Поросенок носит краски.",
                    "Дождь пошел наоборот, // Все зонты унес вперед."
                ],
                "плохие": [
                    "Кот сидит и смотрит. // Это смешно немного.",
                    "Шутка получилась не очень. // Но можно посмеяться.",
                    "Веселая история про собаку. // Она бегала быстро."
                ]
            },
            "философский": {
                "хорошие": [
                    "Что есть жизнь без смысла и цели? // Вечный вопрос в человечьей доле.",
                    "Время течет, как река бесконечная, // Унося с собой все, что вечно.",
                    "Бытие и небытие - две стороны, // Между ними ходим мы, как тени."
                ],
                "плохие": [
                    "Жизнь - это когда живешь. // А смерть - это конец.",
                    "Все в мире сложно устроено. // Надо думать об этом.",
                    "Философия - наука о смысле. // Она изучает жизнь."
                ]
            }
        }
        
        style_data = examples_by_style.get(style, examples_by_style["лирический"])
        
        examples = f"""ПРИМЕРЫ ДЛЯ СТИЛЯ '{style.upper()}':

ХОРОШИЕ ПРИМЕРЫ (что стремиться получить):
"""
        for i, example in enumerate(style_data["хорошие"][:3], 1):
            examples += f"{i}. {example}\n"
        
        examples += "\nАНТИПРИМЕРЫ (чего избегать):\n"
        for i, example in enumerate(style_data["плохие"][:3], 1):
            examples += f"{i}. {example}\n"
        
        examples += f"""

СОВЕТЫ ДЛЯ ЗАПРОСА "{prompt}":
• Используйте слова, связанные с темой
• Поддерживайте единый эмоциональный тон
• Следите за ритмом и рифмой
• Избегайте бытовых, прозаических выражений"""
        
        return examples
    
    def _get_improvement_summary(self, score_diff: float) -> str:
        """Получает текстовое описание улучшения"""
        if score_diff > 0.3:
            return "Значительное улучшение! Стих стал намного лучше."
        elif score_diff > 0.15:
            return "Заметное улучшение. Качество стиха повысилось."
        elif score_diff > 0.05:
            return "Небольшое улучшение. Есть прогресс."
        elif abs(score_diff) <= 0.05:
            return "Изменения незначительны. Стих был уже хорош."
        elif score_diff < -0.05:
            return "Качество немного ухудшилось. Нужно пересмотреть изменения."
        else:
            return "Результат обработки."
    
    def _get_quality_label(self, score: float) -> str:
        """Получает текстовую оценку качества"""
        if score >= 0.9:
            return "Отличное качество ⭐⭐⭐⭐⭐"
        elif score >= 0.8:
            return "Очень хорошее качество ⭐⭐⭐⭐"
        elif score >= 0.7:
            return "Хорошее качество ⭐⭐⭐"
        elif score >= 0.6:
            return "Удовлетворительное качество ⭐⭐"
        elif score >= 0.5:
            return "Приемлемое качество ⭐"
        else:
            return "Требует значительной доработки"
    
    def save_report_to_file(self, report: str, filename: str = None):
        """Сохраняет отчет в файл"""
        import os
        
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.txt"
        
        if not os.path.exists("reports"):
            os.makedirs("reports")
        
        filepath = os.path.join("reports", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return filepath