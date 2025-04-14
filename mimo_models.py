import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------- Basic Blocks for DenseNet -------------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(BasicBlock, self).__init__()
        # BN-ReLU-Conv sequence as mentioned in the paper
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        
        self.bn2 = nn.BatchNorm1d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        return torch.cat([x, out], 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        # Only apply pooling if the feature map is large enough
        self.use_pooling = True
        self.avg_pool = nn.AvgPool1d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        # Check if the sequence length is greater than 1 before pooling
        if out.size(-1) > 1 and self.use_pooling:
            out = self.avg_pool(out)
        return out


class DenseBlock(nn.Module):
    def __init__(self, in_channels, n_layers, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(BasicBlock(in_channels + i * growth_rate, growth_rate))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ------------- DenseNet Model -------------
class DenseNetForMIMO(nn.Module):
    def __init__(self, in_channels=2, rx_antennas=1, tx_antennas=2, num_classes=8, growth_rate=32, block_config=(6, 12, 24, 16)):
        super(DenseNetForMIMO, self).__init__()
        
        # Calculate actual input channels based on tx and rx antennas
        if tx_antennas == 2:
            self.actual_in_channels = in_channels * rx_antennas  # 2 per rx antenna
        elif tx_antennas == 3:
            self.actual_in_channels = 8 * rx_antennas  # 8 per rx antenna
        elif tx_antennas == 4:
            self.actual_in_channels = 4 * rx_antennas  # 4 per rx antenna
        else:
            raise ValueError(f"Unsupported number of transmit antennas: {tx_antennas}")
        
        print(f"Creating DenseNet with rx={rx_antennas}, tx={tx_antennas}, in_channels={in_channels}")
        print(f"Setting actual_in_channels to {self.actual_in_channels}")
        
        # Initial convolution
        self.conv1 = nn.Conv1d(self.actual_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks
        num_features = 64
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        
        for i, num_layers in enumerate(block_config):
            self.dense_blocks.append(DenseBlock(num_features, num_layers, growth_rate))
            num_features += num_layers * growth_rate
            
            if i != len(block_config) - 1:
                self.transition_layers.append(TransitionLayer(num_features, num_features // 2))
                num_features = num_features // 2
        
        # Final batch norm
        self.bn_final = nn.BatchNorm1d(num_features)
        
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, 2, seq_len)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        for i, dense_block in enumerate(self.dense_blocks):
            x = dense_block(x)
            if i < len(self.transition_layers):
                x = self.transition_layers[i](x)
        
        x = self.bn_final(x)
        x = self.relu(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        x = torch.sigmoid(x)  # Binary classification for each bit
        
        return x


# ------------- ResNet Components -------------
class ResidualBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(out_channels * self.expansion)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# ------------- ResNet Model -------------
class ResNetForMIMO(nn.Module):
    def __init__(self, block, num_blocks, in_channels=2, rx_antennas=1, tx_antennas=2, num_classes=8):
        super(ResNetForMIMO, self).__init__()
        
        # Calculate actual input channels based on tx and rx antennas
        if tx_antennas == 2:
            self.actual_in_channels = in_channels * rx_antennas  # 2 per rx antenna
        elif tx_antennas == 3:
            self.actual_in_channels = 8 * rx_antennas  # 8 per rx antenna
        elif tx_antennas == 4:
            self.actual_in_channels = 4 * rx_antennas  # 4 per rx antenna
        else:
            raise ValueError(f"Unsupported number of transmit antennas: {tx_antennas}")
        
        print(f"Creating ResNet with rx={rx_antennas}, tx={tx_antennas}, in_channels={in_channels}")
        print(f"Setting actual_in_channels to {self.actual_in_channels}")
        
        self.in_channels = 64
        
        self.conv1 = nn.Conv1d(self.actual_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = torch.sigmoid(out)  # Binary classification for each bit
        
        return out


def ResNet50ForMIMO(in_channels=2, rx_antennas=1, tx_antennas=2, num_classes=8):
    return ResNetForMIMO(BottleneckBlock, [3, 4, 6, 3], in_channels, rx_antennas, tx_antennas, num_classes)


# ------------- MobileNetV2 Components -------------
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.append(nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise
        layers.extend([
            # Depthwise convolution
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU6(inplace=True),
            
            # Pointwise
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# ------------- MobileNetV2 Model -------------
class MobileNetV2ForMIMO(nn.Module):
    def __init__(self, in_channels=2, rx_antennas=1, tx_antennas=2, num_classes=8, width_mult=1.0):
        super(MobileNetV2ForMIMO, self).__init__()
        
        # Calculate actual input channels based on tx and rx antennas
        if tx_antennas == 2:
            self.actual_in_channels = in_channels * rx_antennas  # 2 per rx antenna
        elif tx_antennas == 3:
            self.actual_in_channels = 8 * rx_antennas  # 8 per rx antenna
        elif tx_antennas == 4:
            self.actual_in_channels = 4 * rx_antennas  # 4 per rx antenna
        else:
            raise ValueError(f"Unsupported number of transmit antennas: {tx_antennas}")
        
        print(f"Creating MobileNetV2 with rx={rx_antennas}, tx={tx_antennas}, in_channels={in_channels}")
        print(f"Setting actual_in_channels to {self.actual_in_channels}")
        
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        
        # Shallow feature extraction part
        self.shallow_features = nn.Sequential(
            nn.Conv1d(self.actual_in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        
        # First inverted residual block
        self.first_block = block(input_channel, 16, 1, 1)
        
        # Building inverted residual blocks (backbone)
        inverted_residual_setting = [
            # t, c, n, s
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Start from 16 channels after first block
        input_channel = 16
        
        self.backbone = nn.ModuleList()
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.backbone.append(block(input_channel, output_channel, stride, t))
                input_channel = output_channel
        
        # Last stage
        self.last_stage = nn.Sequential(
            nn.Conv1d(input_channel, last_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(last_channel),
            nn.ReLU(inplace=True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes),
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Shallow feature extraction
        x = self.shallow_features(x)
        
        # First block
        x = self.first_block(x)
        
        # Backbone
        for layer in self.backbone:
            x = layer(x)
        
        # Last stage
        x = self.last_stage(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        x = torch.sigmoid(x)  # Binary classification for each bit
        
        return x
    

# Add this to your mimo_models.py file

class VGGForMIMO(nn.Module):
    def __init__(self, in_channels=2, rx_antennas=1, tx_antennas=2, num_classes=8):
        super(VGGForMIMO, self).__init__()
        
        # Calculate actual input channels based on tx and rx antennas
        if tx_antennas == 2:
            self.actual_in_channels = in_channels * rx_antennas  # 2 per rx antenna
        elif tx_antennas == 3:
            self.actual_in_channels = 8 * rx_antennas  # 8 per rx antenna
        elif tx_antennas == 4:
            self.actual_in_channels = 4 * rx_antennas  # 4 per rx antenna
        else:
            raise ValueError(f"Unsupported number of transmit antennas: {tx_antennas}")
        
        print(f"Creating VGG with rx={rx_antennas}, tx={tx_antennas}, in_channels={in_channels}")
        print(f"Setting actual_in_channels to {self.actual_in_channels}")
        
        # VGG16-inspired architecture, adapted for 1D signals
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(self.actual_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            
            # No pooling for sequence length of 2
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2, 512),  # 2 is the sequence length
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
            nn.Sigmoid()  # For binary classification
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x