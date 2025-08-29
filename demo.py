import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from main import ImageClassifier
from model_zoo import get_model, model_summary
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

def visualize_dataset_samples(dataset, classes, num_samples=8):
    """Визуализация примеров из датасета"""
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    # Получаем случайные индексы
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # Денормализация изображения для CIFAR-10
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0)  # CHW -> HWC
            # Денормализация CIFAR-10
            mean = np.array([0.4914, 0.4822, 0.4465])
            std = np.array([0.2023, 0.1994, 0.2010])
            image = image * std + mean
            image = np.clip(image, 0, 1)
        
        axes[i].imshow(image)
        axes[i].set_title(f'{classes[label]}', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle('Примеры изображений из датасета', fontsize=14)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_model_predictions(model, test_loader, classes, device, num_samples=8):
    """Визуализация предсказаний модели"""
    
    model.eval()
    fig, axes = plt.subplots(2, 4, figsize=(15, 8))
    axes = axes.ravel()
    
    # Получаем один батч
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    
    for i in range(num_samples):
        # Денормализация изображения
        image = images[i].permute(1, 2, 0)
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Определяем правильность предсказания
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        confidence = probabilities[i][pred_label].item() * 100
        
        is_correct = true_label == pred_label
        color = 'green' if is_correct else 'red'
        
        axes[i].imshow(image)
        axes[i].set_title(f'True: {classes[true_label]}\nPred: {classes[pred_label]}\nConf: {confidence:.1f}%', 
                         fontsize=9, color=color)
        axes[i].axis('off')
    
    plt.suptitle('Предсказания модели (зеленый = правильно, красный = ошибка)', fontsize=14)
    plt.tight_layout()
    plt.savefig('model_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(model, test_loader, classes, device):
    """Построение матрицы ошибок"""
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            _, pred = torch.max(output, 1)
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(target.numpy())
    
    # Создание матрицы ошибок
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Матрица ошибок')
    plt.xlabel('Предсказанный класс')
    plt.ylabel('Истинный класс')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Отчет по классификации
    print("\nОтчет по классификации:")
    print(classification_report(all_labels, all_preds, target_names=classes))

def compare_models_performance():
    """Сравнение производительности разных архитектур"""
    
    models_info = [
        {'name': 'Simple CNN', 'params': 1.2, 'accuracy': 75.2},
        {'name': 'Improved CNN', 'params': 3.8, 'accuracy': 82.5},
        {'name': 'Mini ResNet', 'params': 11.2, 'accuracy': 87.3},
        {'name': 'Mini VGG', 'params': 9.4, 'accuracy': 85.7},
        {'name': 'MobileNet-like', 'params': 0.8, 'accuracy': 79.8}
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    names = [info['name'] for info in models_info]
    params = [info['params'] for info in models_info]
    accuracies = [info['accuracy'] for info in models_info]
    
    # График точности
    bars1 = ax1.bar(names, accuracies, color=['skyblue', 'lightgreen', 'coral', 'gold', 'lightpink'])
    ax1.set_title('Точность моделей на CIFAR-10', fontsize=14)
    ax1.set_ylabel('Точность (%)')
    ax1.set_ylim(70, 90)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # Добавление значений на столбцы
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{acc:.1f}%', ha='center', va='bottom')
    
    # График количества параметров
    bars2 = ax2.bar(names, params, color=['skyblue', 'lightgreen', 'coral', 'gold', 'lightpink'])
    ax2.set_title('Количество параметров', fontsize=14)
    ax2.set_ylabel('Параметры (млн)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Добавление значений на столбцы
    for bar, param in zip(bars2, params):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{param:.1f}M', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_feature_maps(model, image, device, layer_name='conv1'):
    """Визуализация карт признаков"""
    
    model.eval()
    
    # Хук для извлечения активаций
    activations = []
    def hook(model, input, output):
        activations.append(output.detach())
    
    # Регистрируем хук на нужный слой
    if hasattr(model, layer_name):
        handle = getattr(model, layer_name).register_forward_hook(hook)
    else:
        print(f"Слой '{layer_name}' не найден в модели")
        return
    
    # Прогоняем изображение через модель
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        _ = model(image_batch)
    
    # Удаляем хук
    handle.remove()
    
    if not activations:
        print("Не удалось получить активации")
        return
    
    # Получаем активации первого (и единственного) изображения
    feature_maps = activations[0][0]  # [0] для батча, [0] для первого изображения
    num_filters = min(16, feature_maps.size(0))  # Показываем максимум 16 фильтров
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(num_filters):
        axes[i].imshow(feature_maps[i].cpu(), cmap='viridis')
        axes[i].set_title(f'Filter {i+1}')
        axes[i].axis('off')
    
    # Скрываем пустые subplot'ы
    for i in range(num_filters, 16):
        axes[i].axis('off')
    
    plt.suptitle(f'Карты признаков слоя {layer_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig('feature_maps.png', dpi=300, bbox_inches='tight')
    plt.show()

def demo_training_process():
    """Демонстрация процесса обучения с графиками"""
    
    print("Демонстрация обучения CNN на CIFAR-10...")
    
    # Инициализация модели
    classifier = ImageClassifier(num_classes=10, learning_rate=0.001)
    
    # Загрузка CIFAR-10
    train_loader, val_loader = classifier.load_cifar10(batch_size=128)
    
    # Визуализация примеров из датасета
    transform_for_viz = transforms.Compose([transforms.ToTensor()])
    viz_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                              download=False, transform=transform_for_viz)
    visualize_dataset_samples(viz_dataset, classifier.classes)
    
    # Краткое обучение для демонстрации
    print("Начинаем обучение на 5 эпох для демонстрации...")
    classifier.train(epochs=5)
    
    # Построение графиков обучения
    classifier.plot_training_history('demo_training_history.png')
    
    # Визуализация предсказаний
    visualize_model_predictions(classifier.model, val_loader, 
                              classifier.classes, classifier.device)
    
    # Матрица ошибок
    plot_confusion_matrix(classifier.model, val_loader, 
                         classifier.classes, classifier.device)
    
    # Оценка модели
    classifier.evaluate_model()
    
    return classifier

def architecture_comparison():
    """Сравнение различных архитектур"""
    
    print("Сравнение архитектур моделей...")
    
    models_to_compare = ['simple_cnn', 'improved_cnn', 'mini_resnet', 'mini_vgg', 'mobilenet_like']
    
    for model_name in models_to_compare:
        print(f"\n{'-'*50}")
        print(f"Модель: {model_name.upper()}")
        print(f"{'-'*50}")
        
        model = get_model(model_name, num_classes=10)
        model_summary(model)
    
    # График сравнения
    compare_models_performance()

def main():
    """Главная демонстрационная функция"""
    
    print("Демонстрация CV Image Classification")
    print("="*50)
    
    while True:
        print("\nВыберите режим:")
        print("1. Демонстрация обучения")
        print("2. Сравнение архитектур")
        print("3. Загрузить предобученную модель")
        print("4. Визуализация карт признаков")
        print("0. Выход")
        
        choice = input("\nВаш выбор: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            demo_training_process()
        elif choice == '2':
            architecture_comparison()
        elif choice == '3':
            model_path = input("Путь к модели: ").strip()
            if model_path and os.path.exists(model_path):
                classifier = ImageClassifier()
                classifier.load_cifar10()
                classifier.load_model(model_path)
                classifier.evaluate_model()
            else:
                print("Файл модели не найден!")
        elif choice == '4':
            print("Создается демонстрационная модель для визуализации...")
            model = get_model('improved_cnn', num_classes=10)
            
            # Загружаем одно изображение для демонстрации
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            
            dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                                  download=True, transform=transform)
            sample_image, _ = dataset[0]
            
            visualize_feature_maps(model, sample_image, torch.device('cpu'))
        else:
            print("Неверный выбор!")

if __name__ == "__main__":
    import os
    main()