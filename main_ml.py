#!/usr/bin/env python3
"""
Главный файл с GUI чатом для поэтического агента с ML и RL
Интерфейс полностью сохранен как в оригинале
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import queue
import time
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Пробуем импортировать улучшенное ядро
try:
    from agent_core import EnhancedPoetryAgentCore
    AGENT_CLASS = EnhancedPoetryAgentCore
    print("🤖 Загружен агент")
except ImportError as e:
    print(f"⚠️ Не удалось загрузить улучшенного агента: {e}")
    print("🤖 Используется базовый агент")
    from agent_core import PoetryAgentCore
    AGENT_CLASS = PoetryAgentCore

class ChatMessage:
    """Класс для представления сообщения в чате"""
    
    def __init__(self, text, sender, timestamp=None, **kwargs):
        self.text = text
        self.sender = sender
        self.timestamp = timestamp or datetime.now()
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def formatted_time(self):
        return self.timestamp.strftime("%H:%M")

class PoetryAgentGUI:
    """Графический интерфейс чата (интерфейс полностью сохранен)"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🤖 Поэтический агент - Чат-интерфейс")
        self.root.geometry("1000x750")
        self.root.configure(bg='#f0f0f0')
        
        # Инициализируем агента (автоматически выберет версию с ML/RL если доступно)
        self.agent = AGENT_CLASS(verbose=True)
        self.queue = queue.Queue()
        self.processing_thread = None
        self.messages = []
        
        self.should_auto_scroll = True
        
        self.session_stats = {
            "requests": 0,
            "avg_processing_time": 0,
            "start_time": datetime.now()
        }
        
        self.setup_styles()
        self.create_widgets()
        self.process_queue()
        self.input_text.focus_set()
    
    def setup_styles(self):
        self.colors = {
            'bg': '#f0f0f0',
            'user_bg': '#0084ff',
            'agent_bg': '#e4e6eb',
            'user_text': '#ffffff',
            'agent_text': '#050505',
            'status_bg': '#e8f5e8',
            'status_text': '#2e7d32',
            'error_bg': '#ffebee',
            'error_text': '#c62828',
            'input_bg': '#ffffff',
            'border': '#cccccc',
            'button_bg': '#0084ff',
            'button_hover': '#0069d9',
            'timestamp': '#8a8d91'
        }
        
        self.fonts = {
            'normal': ('Segoe UI', 11),
            'bold': ('Segoe UI', 11, 'bold'),
            'mono': ('Consolas', 10),
            'title': ('Segoe UI', 12, 'bold'),
            'message': ('Segoe UI', 11)
        }
    
    def on_mousewheel(self, event):
        self.should_auto_scroll = False
    
    def create_widgets(self):
        main_frame = tk.Frame(self.root, bg=self.colors['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        header_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        title_label = tk.Label(
            header_frame,
            text="🤖 Поэтический агент",
            font=self.fonts['title'],
            bg=self.colors['bg'],
            fg=self.colors['agent_text']
        )
        title_label.pack(side=tk.LEFT)
        
        # Добавляем индикатор ML/RL если доступны
        if hasattr(self.agent, 'ml_enabled') and self.agent.ml_enabled:
            ml_label = tk.Label(
                header_frame,
                text="",
                font=('Segoe UI', 9, 'bold'),
                bg='#4caf50',
                fg='white',
                padx=5,
                pady=2
            )
            ml_label.pack(side=tk.LEFT, padx=(10, 0))
        
        stats_btn = tk.Button(
            header_frame,
            text="📊 Статистика",
            command=self.show_stats,
            bg=self.colors['button_bg'],
            fg='white',
            font=self.fonts['normal'],
            relief=tk.FLAT,
            padx=15,
            cursor='hand2'
        )
        stats_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        clear_btn = tk.Button(
            header_frame,
            text="🗑️ Очистить",
            command=self.clear_chat,
            bg='#757575',
            fg='white',
            font=self.fonts['normal'],
            relief=tk.FLAT,
            padx=15,
            cursor='hand2'
        )
        clear_btn.pack(side=tk.RIGHT)
        
        self.chat_frame = tk.Frame(main_frame, bg='white', relief=tk.SOLID, borderwidth=1)
        self.chat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(
            self.chat_frame,
            wrap=tk.WORD,
            font=self.fonts['message'],
            bg='white',
            relief=tk.FLAT,
            state='disabled',
            height=20
        )
        self.chat_display.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        self.chat_display.bind("<MouseWheel>", self.on_mousewheel)
        
        self.setup_text_tags()
        
        input_frame = tk.Frame(main_frame, bg=self.colors['bg'])
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        input_label = tk.Label(
            input_frame,
            text="Тема стиха:",
            font=self.fonts['bold'],
            bg=self.colors['bg']
        )
        input_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.input_text = tk.Entry(
            input_frame,
            font=self.fonts['normal'],
            bg=self.colors['input_bg'],
            relief=tk.SOLID,
            borderwidth=1
        )
        self.input_text.pack(fill=tk.X, pady=(0, 10))
        self.input_text.bind('<Return>', lambda e: self.send_message())
        
        control_frame = tk.Frame(input_frame, bg=self.colors['bg'])
        control_frame.pack(fill=tk.X)
        
        style_label = tk.Label(
            control_frame,
            text="Стиль:",
            font=self.fonts['normal'],
            bg=self.colors['bg']
        )
        style_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.style_var = tk.StringVar(value="лирический")
        self.style_combo = ttk.Combobox(
            control_frame,
            textvariable=self.style_var,
            values=["лирический", "юмористический", "философский"],
            state="readonly",
            width=15,
            font=self.fonts['normal']
        )
        self.style_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        self.send_button = tk.Button(
            control_frame,
            text="📤 Отправить",
            command=self.send_message,
            bg=self.colors['button_bg'],
            fg='white',
            font=self.fonts['bold'],
            relief=tk.FLAT,
            padx=30,
            pady=8,
            cursor='hand2'
        )
        self.send_button.pack(side=tk.RIGHT)
        
        self.status_label = tk.Label(
            input_frame,
            text="Готов к работе",
            font=self.fonts['normal'],
            bg=self.colors['bg'],
            fg=self.colors['status_text']
        )
        self.status_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.add_welcome_message()
    
    def setup_text_tags(self):
        self.chat_display.tag_config('timestamp', 
            font=('Segoe UI', 9),
            foreground=self.colors['timestamp'],
            spacing1=2,
            spacing3=2
        )
        
        self.chat_display.tag_config('user_message', 
            background=self.colors['user_bg'],
            foreground=self.colors['user_text'],
            lmargin1=10,
            lmargin2=10,
            rmargin=10,
            spacing1=4,
            spacing3=4,
            borderwidth=0,
            relief=tk.FLAT,
            justify=tk.RIGHT,
            font=self.fonts['message']
        )
        
        self.chat_display.tag_config('user_container', 
            background=self.colors['user_bg'],
            foreground=self.colors['user_text'],
            lmargin1=10,
            lmargin2=10,
            rmargin=10,
            spacing1=2,
            spacing3=2,
            borderwidth=0,
            relief=tk.FLAT,
            justify=tk.RIGHT
        )
        
        self.chat_display.tag_config('agent_message', 
            background=self.colors['agent_bg'],
            foreground=self.colors['agent_text'],
            lmargin1=10,
            lmargin2=10,
            rmargin=10,
            spacing1=4,
            spacing3=4,
            borderwidth=0,
            relief=tk.FLAT,
            justify=tk.LEFT,
            font=self.fonts['message']
        )
        
        self.chat_display.tag_config('agent_container', 
            background=self.colors['agent_bg'],
            foreground=self.colors['agent_text'],
            lmargin1=10,
            lmargin2=10,
            rmargin=10,
            spacing1=2,
            spacing3=2,
            borderwidth=0,
            relief=tk.FLAT,
            justify=tk.LEFT
        )
        
        self.chat_display.tag_config('status', 
            background=self.colors['status_bg'],
            foreground=self.colors['status_text'],
            font=self.fonts['normal'],
            justify=tk.CENTER,
            spacing1=3,
            spacing3=3
        )
        
        self.chat_display.tag_config('error', 
            background=self.colors['error_bg'],
            foreground=self.colors['error_text'],
            font=self.fonts['bold'],
            spacing1=3,
            spacing3=3
        )
        
        self.chat_display.tag_config('header', 
            font=self.fonts['bold'],
            spacing1=5,
            spacing3=2
        )
        
        self.chat_display.tag_config('poem', 
            font=self.fonts['mono'],
            background='#f8f8f8',
            lmargin1=10,
            lmargin2=10,
            rmargin=10,
            relief=tk.SOLID,
            borderwidth=1,
            justify=tk.CENTER,
            spacing1=4,
            spacing3=4
        )
    
    def add_welcome_message(self):
        welcome_text = """Добро пожаловать в чат с поэтическим агентом!

🤖 Агент теперь использует современные технологии:
• Нейросетевые модели для оценки стихов
• Reinforcement Learning для улучшения стихов

Как использовать:
1. Введите тему для стиха (например, "зимний вечер")
2. Выберите стиль: лирический, юмористический или философский
3. Нажмите "Отправить" или Enter

Агент выполнит:
• Создание стиха по вашей теме
• Проверку по чек-листу (рифма, размер, стиль, цель)
• Автоматическое улучшение слабых строк
• Предоставление финального варианта и отчета

Попробуйте ввести что-нибудь!"""
        
        self.add_message(welcome_text, 'agent', is_welcome=True)
    
    def add_message(self, text, sender, **kwargs):
        message = ChatMessage(text, sender, **kwargs)
        self.messages.append(message)
        
        self.chat_display.config(state='normal')
        
        if not kwargs.get('is_welcome', False) and sender in ['user', 'agent']:
            self.chat_display.insert(tk.END, f"[{message.formatted_time()}] ", 'timestamp')
        
        if sender == 'user':
            self.chat_display.insert(tk.END, "Вы\n", 'user_container')
            self.chat_display.insert(tk.END, f"{text}\n", 'user_message')
        elif sender == 'agent':
            if kwargs.get('is_welcome'):
                self.chat_display.insert(tk.END, f"{text}\n", 'agent_container')
            else:
                self.chat_display.insert(tk.END, "🤖 Агент\n", 'agent_container')
                self.chat_display.insert(tk.END, f"{text}\n", 'agent_message')
        elif sender == 'status':
            self.chat_display.insert(tk.END, f"⚙️ {text}\n", 'status')
        elif sender == 'error':
            self.chat_display.insert(tk.END, f"❌ {text}\n", 'error')
        
        if sender in ['user', 'agent']:
            self.chat_display.insert(tk.END, "\n", 'timestamp')
        
        self.chat_display.config(state='disabled')
        
        if self.should_auto_scroll:
            self.chat_display.see(tk.END)
        else:
            self.root.after(100, self.check_scroll_position)
    
    def add_poem_message(self, poem, title="Стихотворение"):
        if isinstance(poem, list):
            poem = "\n".join(poem)
        
        self.chat_display.config(state='normal')
        self.chat_display.insert(tk.END, f"📖 {title}:\n", 'header')
        self.chat_display.insert(tk.END, f"{poem}\n\n", 'poem')
        self.chat_display.config(state='disabled')
        
        if self.should_auto_scroll:
            self.chat_display.see(tk.END)
        else:
            self.root.after(100, self.check_scroll_position)
    
    def check_scroll_position(self):
        try:
            current_pos = self.chat_display.yview()
            if current_pos[1] > 0.95:
                self.should_auto_scroll = True
        except:
            pass
    
    def send_message(self):
        prompt = self.input_text.get().strip()
        style = self.style_var.get()
        
        if not prompt:
            messagebox.showwarning("Пустой запрос", "Пожалуйста, введите тему для стиха")
            return
        
        self.add_message(prompt, 'user')
        self.input_text.delete(0, tk.END)
        self.set_input_state(False)
        
        # Показываем статус с учетом ML/RL
        status_text = "🤔 Агент думает"
        if hasattr(self.agent, 'ml_enabled') and self.agent.ml_enabled:
            status_text += ""
        self.status_label.config(text=status_text + "...")
        
        self.processing_thread = threading.Thread(
            target=self.process_request_thread,
            args=(prompt, style),
            daemon=True
        )
        self.processing_thread.start()
        self.session_stats["requests"] += 1
    
    def process_request_thread(self, prompt, style):
        try:
            def callback(event_type, data):
                self.queue.put((event_type, data))
            
            start_time = time.time()
            result = self.agent.process_request(prompt, style, callback=callback)
            processing_time = time.time() - start_time
            
            self.queue.put(('complete', {
                'result': result,
                'processing_time': processing_time
            }))
            
        except Exception as e:
            self.queue.put(('error', f"Ошибка в потоке: {str(e)}"))
    
    def process_queue(self):
        try:
            while True:
                event_type, data = self.queue.get_nowait()
                
                if event_type == 'status':
                    self.add_message(data, 'status')
                elif event_type == 'original_poem':
                    self.add_poem_message(data, "Исходный вариант")
                elif event_type == 'analysis':
                    try:
                        analysis_text = self.agent.format_analysis_display(data)
                        self.add_message(analysis_text, 'agent')
                    except:
                        self.add_message("Анализ завершен", 'agent')
                elif event_type == 'result':
                    self.handle_result(data)
                elif event_type == 'complete':
                    self.handle_completion(data)
                elif event_type == 'error':
                    self.handle_error(data)
                
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)
    
    def handle_result(self, result):
        if result.get('success', False):
            improved_poem = result.get('improved_poem', '')
            if improved_poem:
                self.add_poem_message(improved_poem, "Улучшенный вариант")
            
            improvements = result.get('improvements', [])
            if improvements and improvements[0] != "Улучшения не требуются":
                improvements_text = "✅ Внесенные улучшения:\n"
                for imp in improvements[:5]:
                    improvements_text += f"• {imp}\n"
                self.add_message(improvements_text, 'agent')
            
            score_before = result.get('score_before', 0)
            score_after = result.get('score_after', 0)
            
            # Добавляем информацию о ML оценках если есть
            score_text = f"📈 Оценка улучшилась с {score_before:.2f} до {score_after:.2f}"
            
            if 'ml_score_before' in result and 'ml_score_after' in result:
                ml_improvement = result['ml_score_after'] - result['ml_score_before']
                if ml_improvement > 0:
                    score_text += f" (ML: +{ml_improvement:.2f})"
            
            self.add_message(score_text, 'status')
            
            # Показываем информацию о технологиях
            if result.get('ml_used', False) or result.get('rl_used', False):
                tech_info = "🛠 Использованные технологии: "
                if result.get('ml_used', False):
                    tech_info += "ML "
                if result.get('rl_used', False):
                    tech_info += "RL"
                self.add_message(tech_info, 'status')
    
    def handle_completion(self, data):
        result = data.get('result', {})
        processing_time = data.get('processing_time', 0)
        
        requests = self.session_stats["requests"]
        if requests > 0:
            self.session_stats["avg_processing_time"] = (
                self.session_stats["avg_processing_time"] * (requests - 1) + 
                processing_time
            ) / requests
        
        self.set_input_state(True)
        
        # Показываем информацию о технологиях в статусе
        status_text = f"✅ Готово! Время: {processing_time:.1f} сек"
        if result.get('ml_used', False):
            status_text += " (ML)"
        if result.get('rl_used', False):
            status_text += " (RL)"
        
        self.status_label.config(text=status_text)
        self.add_message("Готов создать еще один стих. Введите новую тему!", 'agent')
    
    def handle_error(self, error_msg):
        self.add_message(error_msg, 'error')
        self.set_input_state(True)
        self.status_label.config(text="❌ Произошла ошибка")
    
    def set_input_state(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.input_text.config(state=state)
        self.style_combo.config(state='readonly' if enabled else tk.DISABLED)
        self.send_button.config(state=state)
        
        if enabled:
            self.input_text.focus_set()
    
    def show_stats(self):
        agent_stats = self.agent.get_stats()
        
        stats_text = f"""📊 Статистика сессии

Общая информация:
• Запросов обработано: {self.session_stats['requests']}
• Среднее время обработки: {self.session_stats['avg_processing_time']:.1f} сек
• Длительность сессии: {self.session_stats['start_time'].strftime('%H:%M')} - {datetime.now().strftime('%H:%M')}

Статистика агента:
• Всего стихов создано: {agent_stats['total_poems']}
• Среднее улучшение оценки: {agent_stats['avg_improvement']:.2f}
• Время работы агента: {agent_stats['session_duration']}"""
        
        # Добавляем ML/RL статистику если есть
        if hasattr(self.agent, 'ml_enabled'):
            stats_text += f"\n\n🤖 ML/RL статистика:"
            stats_text += f"\n• ML доступен: {'Да' if agent_stats.get('ml_enabled', False) else 'Нет'}"
            stats_text += f"\n• RL доступен: {'Да' if agent_stats.get('rl_enabled', False) else 'Нет'}"
            
            if 'ml_predictions' in agent_stats:
                stats_text += f"\n• ML предсказаний: {agent_stats.get('ml_predictions', 0)}"
            
            if 'rl_improvements' in agent_stats:
                stats_text += f"\n• RL улучшений: {agent_stats.get('rl_improvements', 0)}"
            
            if 'avg_ml_score' in agent_stats and agent_stats['avg_ml_score'] > 0:
                stats_text += f"\n• Средняя ML оценка: {agent_stats['avg_ml_score']:.2f}"
            
            if 'avg_rl_reward' in agent_stats and agent_stats['avg_rl_reward'] > 0:
                stats_text += f"\n• Средняя RL награда: {agent_stats['avg_rl_reward']:.2f}"
        
        stats_window = tk.Toplevel(self.root)
        stats_window.title("Статистика")
        stats_window.geometry("500x500")
        stats_window.configure(bg=self.colors['bg'])
        
        stats_display = scrolledtext.ScrolledText(
            stats_window,
            wrap=tk.WORD,
            font=self.fonts['mono'],
            bg='white',
            relief=tk.SOLID,
            borderwidth=1
        )
        stats_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        stats_display.insert(tk.END, stats_text)
        stats_display.config(state='disabled')
        
        # Добавляем кнопку обучения моделей если доступно
        if hasattr(self.agent, 'train_models'):
            def train_models():
                try:
                    result = self.agent.train_models(episodes=50)
                    if 'error' in result:
                        messagebox.showerror("Ошибка", result['error'])
                    else:
                        messagebox.showinfo("Обучение", "Модели успешно обучены!")
                except Exception as e:
                    messagebox.showerror("Ошибка", f"Ошибка обучения: {str(e)}")
            
            train_btn = tk.Button(
                stats_window,
                text="🎯 Обучить модели",
                command=train_models,
                bg='#4caf50',
                fg='white',
                font=self.fonts['normal'],
                padx=15,
                pady=5
            )
            train_btn.pack(pady=(0, 5))
        
        close_btn = tk.Button(
            stats_window,
            text="Закрыть",
            command=stats_window.destroy,
            bg=self.colors['button_bg'],
            fg='white',
            font=self.fonts['normal'],
            padx=20,
            pady=5
        )
        close_btn.pack(pady=(0, 10))
    
    def clear_chat(self):
        if messagebox.askyesno("Очистка чата", "Вы уверены, что хотите очистить историю чата?"):
            self.messages = []
            self.chat_display.config(state='normal')
            self.chat_display.delete(1.0, tk.END)
            self.chat_display.config(state='disabled')
            self.add_welcome_message()
            self.status_label.config(text="Чат очищен")
    
    def on_closing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            if not messagebox.askyesno("Выход", "Агент все еще обрабатывает запрос. Вы уверены, что хотите выйти?"):
                return
        
        self.root.destroy()

def main():
    root = tk.Tk()
    try:
        root.iconbitmap('icon.ico')
    except:
        pass
    
    app = PoetryAgentGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()