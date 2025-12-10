import torch.nn as nn
import torchvision.models as models
from config import NUM_CLASSES, DEVICE

def create_pretrained_alexnet():

    
    #загрузка предобученной AlexNet с лучшими доступными весами
    model = models.alexnet(weights='DEFAULT')
    
    
    print("Загружена предобученная AlexNet")
    print(f"Исходный последний слой: {model.classifier[6]}")
    
    #замена последнего слоя ( ключевой шаг для transfer learning!!!!)
    #исходный слой: 4096 входов -> 1000 выходов (классы ImageNet)
    #новый слой: 4096 входов -> 120 выходов (породы собак)
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    
    print(f"Новый последний слой: {model.classifier[6]}")
    
    #перенос модели на GPU (если доступен)
    model = model.to(DEVICE)
    
    return model

def create_scratch_alexnet():
    """
    СОЗДАНИЕ ALEXNET С СЛУЧАЙНЫМИ ВЕСАМИ (ОБУЧЕНИЕ С НУЛЯ)
    
    Returns:
        model: AlexNet со случайными весами (не обученная)
    """
    
    #загрузка AlexNet БЕЗ предобученных весов (weights=None)
    model = models.alexnet(weights=None)
    
    print("Создана AlexNet со случайными весами (обучение с нуля)")
    print(f"Исходный последний слой: {model.classifier[6]}")
    
    #все равно заменяем последний слой для 120 классов
    model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
    
    print(f"Новый последний слой: {model.classifier[6]}")
    
    #перенос модели на GPU
    model = model.to(DEVICE)
    
    return model