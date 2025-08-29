# CV Image Classification

Проект классификации изображений с использованием сверточных нейронных сетей (CNN) на PyTorch. Реализованы различные архитектуры для обучения на CIFAR-10 и кастомных датасетах.

## Архитектура моделей

### Simple CNN (1.2M параметров)
Базовая двухслойная сверточная сеть:
```
Input(3,32,32) → Conv2d(32) → ReLU → Conv2d(64) → ReLU → MaxPool2d → 
Flatten → Linear(512) → Dropout → Linear(10)
```

### Improved CNN (3.8M параметров) 
Улучшенная версия с Batch Normalization:
```
Input → Conv-Conv-BN-Pool → Conv-Conv-BN-Pool → Conv-BN-Pool → 
FC-Dropout-FC-Dropout-FC
```

### Mini ResNet (11.2M параметров)
Остаточные связи для глубокого обучения:
```
Input → Conv → [ResidualBlock × 6] → AdaptiveAvgPool → Linear
```

### Mini VGG (9.4M параметров)
VGG-подобная архитектура:
```
Input → [VGGBlock] × 4 → AdaptiveAvgPool → FC-Dropout-FC
```

### MobileNet-like (0.8M параметров)
Легкая модель с Depthwise Separable Convolutions:
```
Input → StandardConv → [DepthwiseSeparable] × 6 → GlobalAvgPool → FC
```

## Установка

```bash
# Клонирование и установка зависимостей
git clone <repository>
cd cv-image-classification
python setup.py

# Или ручная установка
pip install -r requirements.txt
```

## Быстрый старт

### Обучение на CIFAR-10

```bash
# Быстрое обучение (10 эпох)
python train.py --mode quick

# Полное обучение конкретной модели
python train.py --mode custom --model mini_resnet --epochs 50 --lr 0.001

# Сравнение всех архитектур
python train.py --mode compare
```

### Основной интерфейс

```bash
# Стандартное обучение
python main.py --epochs 30 --batch-size 128

# Обучение на кастомном датасете
python main.py --dataset custom --train-dir data/train --val-dir data/val

# Предсказание для изображения
python main.py --load-model best_model.pth --predict image.jpg

# Оценка модели
python main.py --load-model best_model.pth --evaluate
```

### Демонстрация возможностей

```bash
python demo.py
```

Интерактивное меню:
- Демонстрация обучения с визуализацией
- Сравнение архитектур
- Визуализация карт признаков
- Анализ предсказаний модели

## Результаты на CIFAR-10

| Модель | Параметры | Точность | Время эпохи (GPU) |
|--------|-----------|----------|-------------------|
| Simple CNN | 1.2M | 75.2% | 15 сек |
| Improved CNN | 3.8M | 82.5% | 30 сек |
| Mini ResNet | 11.2M | 87.3% | 50 сек |
| Mini VGG | 9.4M | 85.7% | 45 сек |
| MobileNet-like | 0.8M | 79.8% | 12 сек |

## Визуализация результатов

### Графики обучения
```python
classifier.plot_training_history('training_plots.png')
```
Создает графики потерь и точности по эпохам.

### Матрица ошибок
```python
from demo import plot_confusion_matrix
plot_confusion_matrix(model, test_loader, classes, device)
```

### Карты признаков
```python
from demo import visualize_feature_maps
visualize_feature_maps(model, sample_image, device, 'conv1')
```

### Анализ предсказаний
```python
from demo import visualize_model_predictions
visualize_model_predictions(model, test_loader, classes, device)
```

## Структура проекта

```
cv-image-classification/
├── main.py                 # Основной класс ImageClassifier
├── model_zoo.py           # Архитектуры моделей
├── train.py               # Упрощенный скрипт обучения
├── demo.py                # Визуализация и анализ
├── setup.py               # Установка проекта
├── requirements.txt       # Зависимости Python
├── README.md             # Документация
├── data/                 # CIFAR-10 и кастомные данные
├── models/               # Сохраненные модели
└── results/              # Графики и результаты
```

## Кастомные данные

### Структура папок
```
data/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── class2/
│       ├── image3.jpg
│       └── image4.jpg
└── val/
    ├── class1/
    └── class2/
```

### Обучение
```bash
python main.py \
    --dataset custom \
    --train-dir data/train \
    --val-dir data/val \
    --epochs 50
```

## API Reference

### ImageClassifier
```python
from main import ImageClassifier

classifier = ImageClassifier(
    num_classes=10,
    learning_rate=0.001
)

# Загрузка CIFAR-10
train_loader, val_loader = classifier.load_cifar10(batch_size=128)

# Обучение
classifier.train(epochs=30)

# Сохранение
classifier.save_model('model.pth')

# Предсказание
predicted_class, confidence = classifier.predict('image.jpg')
```

### Фабрика моделей
```python
from model_zoo import get_model, model_summary

model = get_model('mini_resnet', num_classes=10)
model_summary(model)
```

## Параметры

### Командная строка main.py
- `--epochs` - количество эпох (по умолчанию: 30)
- `--batch-size` - размер батча (128)
- `--lr` - learning rate (0.001)
- `--dataset` - тип данных: cifar10/custom
- `--load-model` - путь к модели
- `--predict` - путь к изображению
- `--evaluate` - режим оценки

### Командная строка train.py
- `--mode` - режим: quick/custom/compare
- `--model` - архитектура модели
- `--epochs` - количество эпох
- `--lr` - learning rate
- `--batch-size` - размер батча

## Производительность

### Системные требования
- Python 3.7+
- 8GB RAM (минимум), 16GB (рекомендуется)
- CUDA GPU (опционально, ускоряет в 5-10 раз)

### Время обучения (30 эпох)
- CPU: 45 мин - 3 часа (зависит от модели)
- GPU: 8-25 мин

### Оптимизация для слабых машин
```bash
# Уменьшение батча
python main.py --batch-size 32

# Простая модель
python train.py --model simple_cnn

# Меньше эпох
python main.py --epochs 10
```

## Технические детали

### Аугментация данных
- RandomHorizontalFlip (p=0.5)
- RandomRotation (10°)
- RandomCrop с padding
- ColorJitter (brightness, contrast, saturation)

### Оптимизация
- Adam optimizer с weight decay (1e-4)
- StepLR scheduler (γ=0.1, step=10)
- Dropout для регуляризации
- Batch Normalization для стабилизации

### Функции потерь
- CrossEntropyLoss для многоклассовой классификации

## Устранение проблем

### Ошибки установки
```bash
# PyTorch с CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU-only версия
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Нехватка памяти
- Уменьшите batch_size до 32 или 16
- Используйте Simple CNN вместо ResNet
- Закройте другие программы

### Медленное обучение
- Проверьте использование GPU: `torch.cuda.is_available()`
- Уменьшите разрешение изображений
- Используйте MobileNet-like архитектуру

## Расширение проекта

### Добавление новой модели
```python
# В model_zoo.py
class NewCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(NewCNN, self).__init__()
        # Ваша архитектура
        
    def forward(self, x):
        # Прямой проход
        return x

# Обновление фабрики
def get_model(model_name, **kwargs):
    models = {
        'new_cnn': NewCNN,
        # ...
    }
```

### Новый датасет
Поместите данные в папки по классам и используйте:
```bash
python main.py --dataset custom --train-dir your_train --val-dir your_val
```

## Лицензия

MIT License
