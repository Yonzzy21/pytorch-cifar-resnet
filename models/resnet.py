'''Customizable ResNet in PyTorch matching course specifications.'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomBlock(nn.Module):
    """A configurable Residual Block.
    
    Contains 2 convolutions with kernel size F x F, followed by BN and ReLU,
    with a skip connection that supports:
      - 'conv': Standard convolution with kernel K x K and stride = stride
      - 'avgpool': AvgPool2d(kernel_size=stride, stride=stride) followed by 1x1 Conv(stride=1)
                   (exact pooling to prevent spatial information loss during downsampling).
    """
    def __init__(self, in_planes, planes, stride=1, conv_kernel=3, skip_kernel=1, shortcut_type='conv'):
        super(CustomBlock, self).__init__()
        
        # Main path: Conv1 (stride = stride) -> BN1 -> ReLU -> Conv2 (stride = 1) -> BN2
        conv1_pad = (conv_kernel - 1) // 2
        self.conv1 = nn.Conv2d(
            in_planes, planes, 
            kernel_size=conv_kernel, 
            stride=stride, 
            padding=conv1_pad, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        
        conv2_pad = (conv_kernel - 1) // 2
        self.conv2 = nn.Conv2d(
            planes, planes, 
            kernel_size=conv_kernel, 
            stride=1, 
            padding=conv2_pad, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Skip connection path:
        # If dimensions and channels match (stride == 1 and in_planes == planes), it is identity.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if shortcut_type == 'avgpool' and stride > 1:
                # Information-preserving downsampling:
                # AvgPool2d pools an exact non-overlapping window of size (stride x stride).
                # Every single input pixel is accounted for (0% discarded pixels).
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm2d(planes)
                )
            else:
                # Standard convolution shortcut
                skip_pad = (skip_kernel - 1) // 2
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, planes, 
                        kernel_size=skip_kernel, 
                        stride=stride, 
                        padding=skip_pad, 
                        bias=False
                    ),
                    nn.BatchNorm2d(planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CustomResNet(nn.Module):
    """Configurable ResNet for 32x32 images.
    
    Parameters:
      num_layers: N (number of residual layers, default 4)
      blocks_per_layer: B_i list (e.g. [2, 2, 2, 2] or [3, 3, 3, 2])
      channels: C_i list (e.g. [40, 80, 160, 320])
      conv_kernel_sizes: F_i list (e.g. [3, 3, 3, 3])
      skip_kernel_sizes: K_i list (e.g. [1, 1, 1, 1] or [3, 3, 3, 3])
      shortcut_type: 'conv' or 'avgpool' (information-preserving downsampling)
      strides: list of strides for each stage (default: [1, 2, 2, 2])
      pool_size: P (average pool kernel size, default 4)
      num_classes: number of output classes (default 10)
    """
    def __init__(self, 
                 num_layers=4,
                 blocks_per_layer=None,
                 channels=None,
                 conv_kernel_sizes=None,
                 skip_kernel_sizes=None,
                 shortcut_type='conv',
                 strides=None,
                 pool_size=4,
                 num_classes=10):
        super(CustomResNet, self).__init__()
        
        # Default hyperparameters
        if blocks_per_layer is None:
            blocks_per_layer = [2, 2, 2, 2]
        if channels is None:
            channels = [64, 128, 256, 512]
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [3] * num_layers
        if skip_kernel_sizes is None:
            skip_kernel_sizes = [1] * num_layers
        if strides is None:
            strides = [1] + [2] * (num_layers - 1)
            
        assert len(blocks_per_layer) == num_layers
        assert len(channels) == num_layers
        assert len(conv_kernel_sizes) == num_layers
        assert len(skip_kernel_sizes) == num_layers
        assert len(strides) == num_layers
        
        self.num_layers = num_layers
        self.pool_size = pool_size
        self.shortcut_type = shortcut_type
        
        # Initial Convolution: 3 -> C_1 channels, 3x3 kernel, stride 1
        initial_channels = channels[0]
        self.conv1 = nn.Conv2d(3, initial_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        
        # Build N residual layers
        self.layers = nn.ModuleList()
        current_in_planes = initial_channels
        
        for i in range(num_layers):
            out_planes = channels[i]
            num_blocks = blocks_per_layer[i]
            conv_k = conv_kernel_sizes[i]
            skip_k = skip_kernel_sizes[i]
            s = strides[i]
            
            layer = self._make_layer(
                in_planes=current_in_planes,
                planes=out_planes,
                num_blocks=num_blocks,
                stride=s,
                conv_kernel=conv_k,
                skip_kernel=skip_k,
                shortcut_type=shortcut_type
            )
            self.layers.append(layer)
            current_in_planes = out_planes
            
        # Classifier Head
        self.linear = nn.Linear(channels[-1], num_classes)

    def _make_layer(self, in_planes, planes, num_blocks, stride, conv_kernel, skip_kernel, shortcut_type):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for s in strides:
            blocks.append(CustomBlock(
                in_planes=in_planes, 
                planes=planes, 
                stride=s, 
                conv_kernel=conv_kernel, 
                skip_kernel=skip_kernel,
                shortcut_type=shortcut_type
            ))
            in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = F.avg_pool2d(out, self.pool_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return CustomResNet()
