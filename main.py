import torch
import torch.nn as nn
import argparse
from data_loader import get_data_loaders
from models import create_pretrained_alexnet, create_scratch_alexnet
from train import train_model
from evaluate import evaluate_model, plot_results, print_final_comparison
from config import DEVICE, LEARNING_RATE, NUM_EPOCHS
import os

def run_experiment1(train_loader, val_loader, test_loader, class_names, results):
    """эксперимент 1: ДООБУЧЕНИЕ ALEXNET С ADAM"""
    print("\n" + "="*50)
    print("1: дообучение AlexNet с Adam")
    print("="*50)
    
    #создание модели
    model = create_pretrained_alexnet()
    
    #оптимизатор Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"создан оптимизатор Adam с learning rate={LEARNING_RATE}")
    
    #обучение модели
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), 
        NUM_EPOCHS, "Fine-Tune AlexNet (Adam)"
    )
    
    #оценка модели на тестовых данных
    precision, recall = evaluate_model(model, test_loader, "Fine-Tune AlexNet (Adam)")
    
    #сохранение результатов
    results["Fine-Tune (Adam)"] = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'precision': precision,
        'recall': recall
    }
    
    #сохранение модели
    os.makedirs('./results', exist_ok=True)
    torch.save(model.state_dict(), './results/fine_tune_adam.pth')
    
    return model, results

def run_experiment2(train_loader, val_loader, test_loader, class_names, results):
    """эксперимент 2: ДООБУЧЕНИЕ ALEXNET С LION (REFINED LION)"""
    print("\n" + "="*50)
    print("2: дообучение AlexNet с Lion")
    print("="*50)
    
    #создание модели
    model = create_pretrained_alexnet()
    
    #попытка использовать Lion оптимизатор
    try:
        from lion_pytorch import Lion
        optimizer = Lion(model.parameters(), lr=LEARNING_RATE)
        print("используется оптимизатор Lion")
    except ImportError:
        #если Lion не установлен, используем AdamW как альтернативу
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        print("Lion не установлен, используется AdamW как альтернатива")
        print("💡 Установите: pip install lion-pytorch")
    
    #обучение модели
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
        NUM_EPOCHS, "Fine-Tune AlexNet (Lion)"
    )
    
    #оценка модели
    precision, recall = evaluate_model(model, test_loader, "Fine-Tune AlexNet (Lion)")
    
    #сохранение результатов
    results["Fine-Tune (Lion)"] = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'precision': precision,
        'recall': recall
    }
    
    #сохранение модели
    os.makedirs('./results', exist_ok=True)
    torch.save(model.state_dict(), './results/fine_tune_lion.pth')
    
    return model, results

def run_experiment3(train_loader, val_loader, test_loader, class_names, results):
    """эксперимент 3: ОБУЧЕНИЕ ALEXNET С НУЛЯ С ADAM"""
    print("\n" + "="*50)
    print("эксперимент 3: Обучение AlexNet с нуля (Adam)")
    print("="*50)
    
    #создание модели со случайными весами
    model = create_scratch_alexnet()
    
    #для обучения с нуля используем больший learning rate
    scratch_lr = LEARNING_RATE * 10
    optimizer = torch.optim.Adam(model.parameters(), lr=scratch_lr)
    print(f"создан оптимизатор Adam с learning rate={scratch_lr} (больше для обучения с нуля)")
    
    #обучение модели
    train_losses, val_accuracies = train_model(
        model, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(),
        NUM_EPOCHS, "Scratch AlexNet (Adam)"
    )
    
    #оценка модели
    precision, recall = evaluate_model(model, test_loader, "Scratch AlexNet (Adam)")
    
    #сохранение результатов
    results["Scratch (Adam)"] = {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'precision': precision,
        'recall': recall
    }
    
    #сохранение модели
    os.makedirs('./results', exist_ok=True)
    torch.save(model.state_dict(), './results/scratch_adam.pth')
    
    return model, results

def load_data():

    print("\n загрузка")
    train_loader, val_loader, test_loader, class_names = get_data_loaders()
    print(f"Загружено {len(class_names)} классов (пород собак)")
    print(f"Размеры выборок: Train={len(train_loader.dataset)}, "
          f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    return train_loader, val_loader, test_loader, class_names

def print_analysis(results):
    """Анализ и вывод результатов"""
    print("\n" + "="*70)
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("="*70)
    
    if not results:
        print("Нет результатов для анализа")
        return
    
    #автоматический анализ результатов
    best_model = max(results.items(), key=lambda x: x[1]['precision'])
    print(f"Лучшая модель по Precision: {best_model[0]} ({best_model[1]['precision']:.4f})")
    
    #сравнение дообучения и обучения с нуля
    if "Fine-Tune (Adam)" in results and "Scratch (Adam)" in results:
        fine_tune_score = results["Fine-Tune (Adam)"]['precision']
        scratch_score = results["Scratch (Adam)"]['precision']
        
        if fine_tune_score > scratch_score:
            print("дообучение работает лучше обучения с нуля")
        else:
            print("обучение с нуля показало сравнимые или лучшие результаты")
    
    #Сравнение оптимизаторов
    if "Fine-Tune (Adam)" in results and "Fine-Tune (Lion)" in results:
        adam_score = results["Fine-Tune (Adam)"]['precision']
        lion_score = results["Fine-Tune (Lion)"]['precision']
        
        if lion_score > adam_score:
            print("Lion показал лучшие результаты чем Adam")
        else:
            print("Adam и Lion показали схожие результаты")

def main():
    
    parser = argparse.ArgumentParser(description='Классификация пород собак с AlexNet')
    parser.add_argument('--experiment', type=int, choices=[1, 2, 3], 
                       help='Номер эксперимента для запуска (1, 2 или 3)')
    parser.add_argument('--all', action='store_true', 
                       help='Запустить все эксперименты')
    
    args = parser.parse_args()
    
    print("="*70)
    print("лаборатная 1: Классификация пород собак")
    print("Архитектура: AlexNet, Оптимизатор: Refined Lion")
    print("="*70)
    
    #папку для результатов, если ее нет
    os.makedirs('./results', exist_ok=True)
    
    #запуск выбранного эксперимента
    if args.all:
        print("\n ЗАПУСК ВСЕХ ЭКСПЕРИМЕНТОВ")
        train_loader, val_loader, test_loader, class_names = load_data()
        results = {}
        
        run_experiment1(train_loader, val_loader, test_loader, class_names, results)
        run_experiment2(train_loader, val_loader, test_loader, class_names, results)
        run_experiment3(train_loader, val_loader, test_loader, class_names, results)
        
        #визуализация результатов
        print("\n" + "="*50)
        print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
        print("="*50)
        plot_results(results)
        print_final_comparison(results)
        print_analysis(results)
        
    elif args.experiment:
        print(f"\n ЗАПУСК ЭКСПЕРИМЕНТА {args.experiment}")
        train_loader, val_loader, test_loader, class_names = load_data()
        results = {}
        
        if args.experiment == 1:
            model, results = run_experiment1(train_loader, val_loader, test_loader, class_names, results)
        elif args.experiment == 2:
            model, results = run_experiment2(train_loader, val_loader, test_loader, class_names, results)
        elif args.experiment == 3:
            model, results = run_experiment3(train_loader, val_loader, test_loader, class_names, results)
        
        print_analysis(results)
        
    else:
        print("\n  Использование:")
        print(" python main.py --all    # Запустить все эксперименты")
        print(" python main.py --experiment 1 # Запустить только эксперимент 1")
        print(" python main.py --experiment 2 # Запустить только эксперимент 2")
        print(" python main.py --experiment 3  # Запустить только эксперимент 3")
        print("\n Результаты сохраняются в папке ./results/")
    
    print("\n ВЫПОЛНЕНИЕ ЗАВЕРШЕНО")

# Точка входа в программу
if __name__ == "__main__":
    main()