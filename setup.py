#!/usr/bin/env python3
"""
Установка и настройка CV Image Classification проекта
"""

import subprocess
import sys
import os
from pathlib import Path
import torch

def check_python_version():
    """Проверка версии Python"""
    if sys.version_info < (3, 7):
        print("Требуется Python 3.7 или новее")
        return False
    
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def install_requirements():
    """Установка зависимостей"""
    print("Установка зависимостей...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Зависимости установлены успешно!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка установки зависимостей: {e}")
        return False

def check_torch_installation():
    """Проверка установки PyTorch"""
    try:
        import torch
        import torchvision
        
        print(f"PyTorch {torch.__version__}")
        print(f"torchvision {torchvision.__version__}")
        
        # Проверка CUDA
        if torch.cuda.is_available():
            print(f"CUDA доступна: {torch.cuda.get_device_name(0)}")
            print(f"   Версия CUDA: {torch.version.cuda}")
        else:
            print(" CUDA недоступна, будет использоваться CPU")
        
        return True
    except ImportError as e:
        print(f"Проблема с PyTorch: {e}")
        return False

def create_directories():
    """Создание необходимых директорий"""
    print("Создание директорий...")
    
    directories = [
        'data',
        'models',
        'results',
        'checkpoints'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  {directory}/")

def download_cifar10():
    """Предварительная загрузка CIFAR-10"""
    print("Загрузка CIFAR-10...")
    
    try:
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        
        # Простая трансформация для загрузки
        transform = transforms.Compose([transforms.ToTensor()])
        
        # Загружаем train и test наборы
        train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                        download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                       download=True, transform=transform)
        
        print(f"CIFAR-10 загружен: {len(train_dataset)} train, {len(test_dataset)} test")
        return True
        
    except Exception as e:
        print(f"Ошибка загрузки CIFAR-10: {e}")
        return False

def test_model_creation():
    """Тест создания моделей"""
    print("Тестирование создания моделей...")
    
    try:
        from model_zoo import get_model, model_summary
        
        models_to_test = ['simple_cnn', 'improved_cnn', 'mini_resnet']
        
        for model_name in models_to_test:
            try:
                model = get_model(model_name, num_classes=10)
                print(f"  {model_name}: {sum(p.numel() for p in model.parameters()):,} параметров")
            except Exception as e:
                print(f"  {model_name}: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Ошибка тестирования моделей: {e}")
        return False

def test_training_pipeline():
    """Тест пайплайна обучения"""
    print("Тестирование пайплайна обучения...")
    
    try:
        from main import ImageClassifier
        import torch
        
        # Создаем мини-классификатор для теста
        classifier = ImageClassifier(num_classes=10, learning_rate=0.01)
        
        # Создаем фиктивные данные
        dummy_data = torch.randn(4, 3, 32, 32)
        dummy_target = torch.randint(0, 10, (4,))
        
        # Тест прямого прохода
        output = classifier.model(dummy_data)
        assert output.shape == (4, 10), "Неправильная размерность выхода"
        
        # Тест функции потерь
        loss = classifier.criterion(output, dummy_target)
        assert loss.item() > 0, "Функция потерь работает неправильно"
        
        print("Пайплайн обучения работает корректно")
        return True
        
    except Exception as e:
        print(f"Ошибка тестирования пайплайна: {e}")
        return False

def create_example_script():
    """Создание примера использования"""
    
    example_code = '''#!/usr/bin/env python3
"""
Пример быстрого запуска CV Image Classification
"""

from main import ImageClassifier

def main():
    print("Быстрый пример обучения на CIFAR-10")
    
    # Создание классификатора
    classifier = ImageClassifier(num_classes=10, learning_rate=0.001)
    
    # Загрузка CIFAR-10
    train_loader, val_loader = classifier.load_cifar10(batch_size=64)
    
    # Обучение на 5 эпохах для демонстрации
    classifier.train(epochs=5)
    
    # Сохранение модели
    classifier.save_model('example_model.pth')
    
    # Графики обучения
    classifier.plot_training_history('example_training.png')
    
    # Оценка модели
    classifier.evaluate_model()
    
    print("Пример завершен! Проверьте файлы:")
    print("- example_model.pth (сохраненная модель)")
    print("- example_training.png (графики обучения)")

if __name__ == "__main__":
    main()
'''
    
    with open('example.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("Создан example.py для быстрого старта")

def print_usage_guide():
    """Вывод руководства по использованию"""
    
    print("\n" + "="*60)
    print("Установка завершена успешно!")
    print("="*60)
    
    print("\nБыстрый старт:")
    print("1. python example.py              # Быстрый пример")
    print("2. python train.py --mode quick   # Обучение по умолчанию")
    print("3. python demo.py                 # Интерактивная демонстрация")
    
    print("\nОсновные команды:")
    print("# Полное обучение")
    print("python main.py --epochs 30 --batch-size 128")
    
    print("\n# Сравнение моделей")
    print("python train.py --mode compare")
    
    print("\n# Обучение конкретной архитектуры")
    print("python train.py --mode custom --model mini_resnet --epochs 50")
    
    print("\n# Предсказание для изображения")
    print("python main.py --load-model best_model.pth --predict image.jpg")
    
    print("\nСтруктура проекта:")
    print("- data/          # Датасеты (CIFAR-10 автоматически загружен)")
    print("- models/        # Сохраненные модели")  
    print("- results/       # Графики и результаты")
    print("- checkpoints/   # Промежуточные checkpoint'ы")
    
    print("\nПолезные ресурсы:")
    print("- README.md      # Подробная документация")
    print("- model_zoo.py   # Все доступные архитектуры")
    print("- demo.py        # Примеры визуализации")

def main():
    """Главная функция установки"""
    
    print("Установка CV Image Classification Project")
    print("="*50)
    
    success = True
    
    # Проверки
    success &= check_python_version()
    success &= install_requirements()
    success &= check_torch_installation()
    
    if not success:
        print("\nУстановка прервана из-за ошибок")
        return False
    
    # Настройка
    create_directories()
    
    # Загрузка данных
    success &= download_cifar10()
    
    # Тестирование
    success &= test_model_creation()
    success &= test_training_pipeline()
    
    # Создание примеров
    create_example_script()
    
    if success:
        print_usage_guide()
    else:
        print("\nУстановка завершена с предупреждениями")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)