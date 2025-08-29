import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    """Простая CNN для начинающих"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Для CIFAR-10 (32x32 изображения)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class ImprovedCNN(nn.Module):
    """Улучшенная CNN с Batch Normalization и дополнительными слоями"""
    
    def __init__(self, num_classes=10, input_channels=3, dropout_rate=0.5):
        super(ImprovedCNN, self).__init__()
        
        # Первый блок
        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Второй блок
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Третий блок
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
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
        x = self.bn3(F.relu(self.conv5(x)))
        x = self.pool3(x)
        
        # Полносвязные слои
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

class ResidualBlock(nn.Module):
    """Резидуальный блок для ResNet"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()
            
    def forward(self, x):
        identity = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out

class MiniResNet(nn.Module):
    """Упрощенная версия ResNet для CIFAR-10"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(MiniResNet, self).__init__()
        
        # Начальный слой
        self.conv1 = nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # Резидуальные блоки
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        # Классификатор
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class VGGBlock(nn.Module):
    """VGG блок с несколькими сверточными слоями"""
    
    def __init__(self, in_channels, out_channels, num_convs):
        super(VGGBlock, self).__init__()
        
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, 
                                  out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class MiniVGG(nn.Module):
    """Упрощенная версия VGG для CIFAR-10"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(MiniVGG, self).__init__()
        
        self.features = nn.Sequential(
            VGGBlock(input_channels, 64, 2),   # 32x32 -> 16x16
            VGGBlock(64, 128, 2),              # 16x16 -> 8x8
            VGGBlock(128, 256, 3),             # 8x8 -> 4x4
            VGGBlock(256, 512, 3),             # 4x4 -> 2x2
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    """Депthwise Separable Convolution для эффективных моделей"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # Depthwise convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise convolution
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

class MobileNetLike(nn.Module):
    """Легкая модель в стиле MobileNet для CIFAR-10"""
    
    def __init__(self, num_classes=10, input_channels=3):
        super(MobileNetLike, self).__init__()
        
        # Стандартная свертка
        self.conv1 = nn.Conv2d(input_channels, 32, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Depthwise Separable блоки
        self.layers = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=1),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 128, stride=1),
            DepthwiseSeparableConv(128, 256, stride=2),
            DepthwiseSeparableConv(256, 256, stride=1),
            DepthwiseSeparableConv(256, 512, stride=2),
        )
        
        # Классификатор
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(x)
        x = self.classifier(x)
        return x

def get_model(model_name, num_classes=10, input_channels=3, **kwargs):
    """Фабрика моделей"""
    
    models = {
        'simple_cnn': SimpleCNN,
        'improved_cnn': ImprovedCNN,
        'mini_resnet': MiniResNet,
        'mini_vgg': MiniVGG,
        'mobilenet_like': MobileNetLike
    }
    
    if model_name not in models:
        raise ValueError(f"Модель '{model_name}' не найдена. Доступные: {list(models.keys())}")
    
    model_class = models[model_name]
    
    # Передаем дополнительные параметры если модель их поддерживает
    try:
        return model_class(num_classes=num_classes, input_channels=input_channels, **kwargs)
    except TypeError:
        # Если модель не поддерживает дополнительные параметры
        return model_class(num_classes=num_classes, input_channels=input_channels)

def count_parameters(model):
    """Подсчет количества параметров модели"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {'trainable': trainable, 'total': total}

def model_summary(model, input_size=(1, 3, 32, 32)):
    """Краткая информация о модели"""
    model.eval()
    
    # Подсчет параметров
    params = count_parameters(model)
    
    # Тестовый прогон для получения размеров
    with torch.no_grad():
        x = torch.randn(input_size)
        output = model(x)
    
    print(f"Модель: {model.__class__.__name__}")
    print(f"Параметры: {params['trainable']:,} обучаемых, {params['total']:,} всего")
    print(f"Вход: {input_size}")
    print(f"Выход: {output.shape}")
    print(f"Размер модели: {sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    # Тестирование всех моделей
    models_to_test = ['simple_cnn', 'improved_cnn', 'mini_resnet', 'mini_vgg', 'mobilenet_like']
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        model = get_model(model_name, num_classes=10)
        model_summary(model)