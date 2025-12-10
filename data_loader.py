import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
import numpy as np
import os
from config import DATA_PATH, BATCH_SIZE

def get_data_loaders():

    #преобразования для изображений 
    transform = transforms.Compose([

        #alexNet требует изображения размером 224x224 пикселя
        transforms.Resize(256),            
        transforms.CenterCrop(224),        
        transforms.ToTensor(),    

        #нормализация по каналам RGB (значения из ImageNet)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    #существование пути к данным
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Путь к данным не найден: {DATA_PATH}")
    
    dataset = ImageFolder(DATA_PATH, transform=transform)  #загрузка датасета
    
    print(f"Загружен датасет: {len(dataset)} изображений, {len(dataset.classes)} классов")
    
    #метки изображений
    labels = np.array([label for _, label in dataset.samples])
    
    #стратифицированное разделение на train и temp (val+test)
    train_indices, temp_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.3,  #30% для val+test
        random_state=42,
        stratify=labels  #сохраняем пропорции классов
    )
    
    #разделение temp на validation и test
    temp_labels = labels[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,  #15% test, 15% val (от общего количества)
        random_state=42,
        stratify=temp_labels  # сохранение
    )
    
    #подмножества с помощью Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    #распределение классов
    print("\n Проверка распределения классов:")
    print(f"Всего классов: {len(dataset.classes)}")
    
    #Подсчет классов в подмножестве
    def count_classes(subset_indices):
        class_counts = {}
        for idx in subset_indices:
            _, label = dataset.samples[idx]
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
    
    train_counts = count_classes(train_indices)
    val_counts = count_classes(val_indices)
    test_counts = count_classes(test_indices)
    
    print(f"Train set: {len(train_indices)} samples, {len(train_counts)} классов представлено")
    print(f"Val set: {len(val_indices)} samples, {len(val_counts)} классов представлено")
    print(f"Test set: {len(test_indices)} samples, {len(test_counts)} классов представлено")
    
    #проверка все классы представлены во всех наборах
    all_classes_present = True
    for i in range(len(dataset.classes)):
        if i not in train_counts or i not in val_counts or i not in test_counts:
            all_classes_present = False
            print(f"⚠️  Класс {i} ({dataset.classes[i]}) отсутствует в одном из наборов!")
            break
    
    if all_classes_present:
        print("все классы представлены во всех наборах данных")
    
    #показываем распределение для первых 3 классов
    print("\n Пример распределения по первым 3 классам:")
    for i, class_name in enumerate(dataset.classes[:3]):
        train_count = train_counts.get(i, 0)
        val_count = val_counts.get(i, 0)
        test_count = test_counts.get(i, 0)
        total_count = train_count + val_count + test_count
        
        print(f"{class_name}: Всего={total_count}, "
              f"Train={train_count} ({train_count/total_count*100:.1f}%), "
              f"Val={val_count} ({val_count/total_count*100:.1f}%), "
              f"Test={test_count} ({test_count/total_count*100:.1f}%)")
    
    #DATA LOADER
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\n✅ DataLoader созданы успешно")
    
    return train_loader, val_loader, test_loader, dataset.classes