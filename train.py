#!/usr/bin/env python3
"""
Упрощенный скрипт для быстрого обучения CNN на CIFAR-10
"""

import torch
from main import ImageClassifier
from model_zoo import get_model
import argparse
import os

def quick_train():
    """Быстрое обучение с настройками по умолчанию"""
    
    print("Быстрое обучение CNN на CIFAR-10")
    print("="*40)
    
    # Инициализация
    classifier = ImageClassifier(num_classes=10, learning_rate=0.001)
    
    # Загрузка данных
    print("Загрузка CIFAR-10...")
    train_loader, val_loader = classifier.load_cifar10(batch_size=128)
    
    # Обучение
    print("Начинаем обучение...")
    classifier.train(epochs=10)
    
    # Сохранение
    classifier.save_model('trained_model.pth')
    
    # Графики
    classifier.plot_training_history('training_results.png')
    
    # Оценка
    print("\nФинальная оценка:")
    classifier.evaluate_model()
    
    print(f"\nМодель сохранена в trained_model.pth")
    print(f"Графики сохранены в training_results.png")

def train_custom_model(model_name, epochs, lr, batch_size):
    """Обучение кастомной модели"""
    
    print(f"Обучение модели {model_name}")
    print(f"Эпох: {epochs}, LR: {lr}, Batch size: {batch_size}")
    
    # Создаем кастомную модель
    model = get_model(model_name, num_classes=10)
    
    # Заменяем модель в классификаторе
    classifier = ImageClassifier(num_classes=10, learning_rate=lr)
    classifier.model = model.to(classifier.device)
    
    # Обновляем оптимизатор для новой модели
    classifier.optimizer = torch.optim.Adam(classifier.model.parameters(), lr=lr, weight_decay=1e-4)
    classifier.scheduler = torch.optim.lr_scheduler.StepLR(classifier.optimizer, step_size=10, gamma=0.1)
    
    # Загрузка данных
    train_loader, val_loader = classifier.load_cifar10(batch_size=batch_size)
    
    # Обучение
    classifier.train(epochs=epochs)
    
    # Сохранение
    model_path = f'{model_name}_trained.pth'
    classifier.save_model(model_path)
    
    # Графики
    plots_path = f'{model_name}_training.png'
    classifier.plot_training_history(plots_path)
    
    # Оценка
    classifier.evaluate_model()
    
    print(f"Модель {model_name} сохранена в {model_path}")
    print(f"Графики сохранены в {plots_path}")

def compare_all_models():
    """Сравнение всех доступных архитектур"""
    
    models = ['simple_cnn', 'improved_cnn', 'mini_resnet', 'mini_vgg', 'mobilenet_like']
    results = {}
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Обучение {model_name}")
        print(f"{'='*50}")
        
        try:
            # Быстрое обучение на 5 эпохах для сравнения
            classifier = ImageClassifier(num_classes=10, learning_rate=0.001)
            
            # Замена модели
            model = get_model(model_name, num_classes=10)
            classifier.model = model.to(classifier.device)
            classifier.optimizer = torch.optim.Adam(classifier.model.parameters(), lr=0.001, weight_decay=1e-4)
            
            # Загрузка данных
            train_loader, val_loader = classifier.load_cifar10(batch_size=64)  # Меньший батч для скорости
            
            # Короткое обучение
            classifier.train(epochs=5)
            
            # Оценка
            accuracy = classifier.evaluate_model()
            results[model_name] = accuracy
            
            # Сохранение
            classifier.save_model(f'{model_name}_comparison.pth')
            
        except Exception as e:
            print(f"Ошибка при обучении {model_name}: {e}")
            results[model_name] = 0.0
    
    # Вывод результатов сравнения
    print(f"\n{'='*60}")
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МОДЕЛЕЙ (5 эпох)")
    print(f"{'='*60}")
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for model_name, accuracy in sorted_results:
        print(f"{model_name:20}: {accuracy:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='Обучение CNN для классификации изображений')
    parser.add_argument('--mode', choices=['quick', 'custom', 'compare'], default='quick',
                        help='Режим обучения')
    parser.add_argument('--model', type=str, default='improved_cnn',
                        help='Архитектура модели')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Количество эпох')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Размер батча')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_train()
    elif args.mode == 'custom':
        train_custom_model(args.model, args.epochs, args.lr, args.batch_size)
    elif args.mode == 'compare':
        compare_all_models()

if __name__ == "__main__":
    main()