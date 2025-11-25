import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

class PneumoniaCNN(nn.Module):
    """
    A 4-layer CNN for binary classification (Pneumonia vs Normal)
    """
    def __init__(self, in_channels=1, num_classes=2):
        super(PneumoniaCNN, self).__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.layer1 = conv_block(in_channels, 32)
        self.layer2 = conv_block(32, 64)
        self.layer3 = conv_block(64, 128)
        self.layer4 = conv_block(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


class PneumoniaResNet(nn.Module):
    """
    Transfer learning model using pretrained ResNet18 from torchvision.
    """
    def __init__(self, num_classes=2, freeze_backbone=True):
        super(PneumoniaResNet, self).__init__()
     
        self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        # Convert first conv layer to accept grayscale (1 channel)
        self.resnet.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

    
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

  
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)