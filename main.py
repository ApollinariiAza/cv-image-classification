import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from datetime import datetime
import json
from pathlib import Path
from PIL import Image
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNN(nn.Module):
    """Сверточная нейронная сеть для классификации изображений"""
    
    def __init__(self, num_classes=10, input_channels=3, dropout_rate=0.5):
        super(CNN, self).__init__()
        
        # Первый блок сверток
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Второй блок сверток
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Третий блок сверток
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(256 * 4 * 4, 512)  # Для CIFAR-10 (32x32 -> 4x4 после 3 pooling)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Первый блок
        x = F.relu(self.conv1(x))
        x = self.bn1(F.relu(self.conv2(x)))
        x = self.pool1(x)
        
        # Второй блок
        x = F.relu(self.conv3(x))
        x = self.bn2(F.relu(self.conv4(x)))
        x = self.pool2(x)
        
        # Третий блок
        x = F.relu(self.conv5(x))
        x = self.bn3(F.relu(self.conv6(x)))
        x = self.pool3(x)
        
        # Flatten для полносвязных слоев
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class CustomDataset(Dataset):
    """Кастомный датасет для загрузки изображений из папок"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        # Сбор всех путей к изображениям
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[class_name]))
            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, self.class_to_idx[class_name]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class ImageClassifier:
    """Класс для обучения и оценки модели классификации изображений"""
    
    def __init__(self, num_classes=10, learning_rate=0.001, device=None):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Инициализация модели
        self.model = CNN(num_classes=num_classes)
        self.model.to(self.device)
        
        # Оптимизатор и функция потерь
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Для отслеживания метрик
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        logger.info(f"Модель инициализирована на устройстве: {self.device}")
        logger.info(f"Количество параметров: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def load_cifar10(self, batch_size=128, data_dir='./data'):
        """Загружает датасет CIFAR-10"""
        
        # Аугментация для обучающего набора
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Нормализация для валидационного набора
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # Загрузка CIFAR-10
        train_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform
        )
        
        val_dataset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=val_transform
        )
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Классы CIFAR-10
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        logger.info(f"CIFAR-10 загружен: {len(train_dataset)} обучающих, {len(val_dataset)} валидационных изображений")
        
        return self.train_loader, self.val_loader
    
    def load_custom_dataset(self, train_dir, val_dir, batch_size=128, img_size=32):
        """Загружает кастомный датасет из папок"""
        
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        train_dataset = CustomDataset(train_dir, transform=train_transform)
        val_dataset = CustomDataset(val_dir, transform=val_transform)
        
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        self.classes = train_dataset.classes
        logger.info(f"Кастомный датасет загружен: {len(train_dataset)} обучающих, {len(val_dataset)} валидационных изображений")
        logger.info(f"Классы: {self.classes}")
        
        return self.train_loader, self.val_loader
    
    def train_epoch(self):
        """Обучение одной эпохи"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                logger.info(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100.0 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs=50):
        """Полное обучение модели"""
        logger.info(f"Начинаем обучение на {epochs} эпох...")
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            logger.info(f'\nЭпоха {epoch+1}/{epochs}')
            
            # Обучение
            train_loss, train_acc = self.train_epoch()
            
            # Валидация
            val_loss, val_acc = self.validate()
            
            # Обновление learning rate
            self.scheduler.step()
            
            # Сохранение метрик
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            logger.info(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            logger.info(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('best_model.pth', epoch, val_acc)
                logger.info(f'Новая лучшая модель сохранена! Точность: {val_acc:.2f}%')
    
    def save_model(self, filename, epoch=None, accuracy=None):
        """Сохранение модели и метрик"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'classes': self.classes,
            'num_classes': self.num_classes,
            'epoch': epoch,
            'accuracy': accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, filename)
        logger.info(f"Модель сохранена в {filename}")
    
    def load_model(self, filename):
        """Загрузка модели"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Восстановление метрик если есть
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.train_accuracies = checkpoint['train_accuracies']
            self.val_losses = checkpoint['val_losses']
            self.val_accuracies = checkpoint['val_accuracies']
        
        if 'classes' in checkpoint:
            self.classes = checkpoint['classes']
            
        logger.info(f"Модель загружена из {filename}")
    
    def plot_training_history(self, save_path='training_plots.png'):
        """Построение графиков обучения"""
        if not self.train_losses:
            logger.warning("Нет данных для построения графиков")
            return
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss', fontsize=16)
        ax1.set_xlabel('Epoch', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # График точности
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy', fontsize=16)
        ax2.set_xlabel('Epoch', fontsize=14)
        ax2.set_ylabel('Accuracy (%)', fontsize=14)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Графики сохранены в {save_path}")
    
    def predict(self, image_path):
        """Предсказание для одного изображения"""
        self.model.eval()
        
        # Загружаем и преобразуем изображение
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probabilities = F.softmax(output, dim=1)
            _, predicted = output.max(1)
            
        predicted_class = self.classes[predicted.item()]
        confidence = probabilities[0][predicted].item() * 100
        
        return predicted_class, confidence
    
    def evaluate_model(self):
        """Детальная оценка модели"""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = list(0. for i in range(self.num_classes))
        class_total = list(0. for i in range(self.num_classes))
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # По классам
                c = predicted.eq(target)
                for i in range(target.size(0)):
                    label = target[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
        
        overall_accuracy = 100.0 * correct / total
        
        logger.info(f"\nОбщая точность модели: {overall_accuracy:.2f}%")
        logger.info("Точность по классам:")
        
        for i in range(self.num_classes):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                logger.info(f"  {self.classes[i]}: {acc:.2f}%")
        
        return overall_accuracy

def main():
    parser = argparse.ArgumentParser(description='CNN для классификации изображений')
    parser.add_argument('--dataset', choices=['cifar10', 'custom'], default='cifar10',
                        help='Тип датасета')
    parser.add_argument('--train-dir', type=str, help='Путь к обучающим данным (для custom)')
    parser.add_argument('--val-dir', type=str, help='Путь к валидационным данным (для custom)')
    parser.add_argument('--epochs', type=int, default=30, help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=128, help='Размер батча')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--load-model', type=str, help='Загрузить предобученную модель')
    parser.add_argument('--predict', type=str, help='Путь к изображению для предсказания')
    parser.add_argument('--evaluate', action='store_true', help='Оценить загруженную модель')
    
    args = parser.parse_args()
    
    # Инициализация классификатора
    classifier = ImageClassifier(learning_rate=args.lr)
    
    # Загрузка данных
    if args.dataset == 'cifar10':
        classifier.load_cifar10(batch_size=args.batch_size)
    elif args.dataset == 'custom':
        if not args.train_dir or not args.val_dir:
            logger.error("Для custom датасета нужны --train-dir и --val-dir")
            return
        classifier.load_custom_dataset(args.train_dir, args.val_dir, batch_size=args.batch_size)
    
    # Загрузка модели если указано
    if args.load_model:
        classifier.load_model(args.load_model)
    
    # Режим предсказания
    if args.predict:
        if not args.load_model:
            logger.error("Для предсказания нужна загруженная модель (--load-model)")
            return
        
        predicted_class, confidence = classifier.predict(args.predict)
        logger.info(f"Предсказание: {predicted_class} (уверенность: {confidence:.2f}%)")
        return
    
    # Режим оценки
    if args.evaluate:
        if not args.load_model:
            logger.error("Для оценки нужна загруженная модель (--load-model)")
            return
        
        classifier.evaluate_model()
        return
    
    # Обучение
    if not args.load_model:
        classifier.train(epochs=args.epochs)
        classifier.save_model('final_model.pth')
    
    # Построение графиков
    classifier.plot_training_history('training_history.png')
    
    # Финальная оценка
    classifier.evaluate_model()

if __name__ == "__main__":
    main()