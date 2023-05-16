#### adapted directly from:
# https://github.com/danielajisafe/Audio_WaveForm_Paper_Implementation/blob/master/Combined_Group_Audio_Waveform_Paper_Implementation.ipynb
#### the architectures are implementations for the presented in:
# https://arxiv.org/pdf/1610.00087.pdf

import torch
import torch.nn as nn
from .utils import make_layer
import torch.nn.functional as F
class TabularRes2DCNN(nn.Module):
    def __init__(self, act_func, dropout, channels_hidden_dim, num_classes, num_cnn_layers, kernel_size, stride,
                 use_batchnorm, pool_size):
        super(TabularRes2DCNN, self).__init__()

        self.act_func = make_layer(act_func)
        self.dropout = dropout
        self.channels_hidden_dim = channels_hidden_dim
        self.num_cnn_layers = num_cnn_layers
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_batchnorm = use_batchnorm
        self.pool_size = pool_size

        # Define the convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_cnn_layers):
            if i == 0:
                conv_layer = nn.Conv2d(in_channels=1, out_channels=channels_hidden_dim, kernel_size=kernel_size,
                                       stride=stride)
            else:
                conv_layer = nn.Conv2d(in_channels=channels_hidden_dim, out_channels=channels_hidden_dim, kernel_size=kernel_size,
                                       stride=stride)
            self.conv_layers.append(conv_layer)

            # Add batch normalization
            if self.use_batchnorm:
                bn_layer = make_layer('BatchNorm2d', channels_hidden_dim)
                self.conv_layers.append(bn_layer)

            # Add max-pooling layer
            pool_layer = make_layer('MaxPool2d', self.pool_size)
            self.conv_layers.append(pool_layer)

        # Define the fully connected layers
        fc_layer = nn.Linear(channels_hidden_dim, num_classes)
        self.fc = fc_layer

    def forward(self, x):

        # Save the original spatial dimensions
        original_spatial_dim_h = x.size(2)
        original_spatial_dim_w = x.size(3)

        # Apply the convolutional layers
        for i, conv in enumerate(self.conv_layers):
            if i % 3 == 0:
                # Apply the convolutional layer
                x = self.act_func(conv(x))
            elif i % 3 == 1 and self.use_batchnorm:
                # Apply the batch normalization layer
                x = self.conv_layers[i](x)
            elif i % 3 == 2:
                # Apply the max-pooling layer
                x = self.conv_layers[i](x)

            # Pad the output to preserve the spatial dimensions
            pad_size_h = original_spatial_dim_h - x.size(2)
            pad_size_w = original_spatial_dim_w - x.size(3)
            if pad_size_h > 0 or pad_size_w > 0:
                x = nn.functional.pad(x, (0, pad_size_w, 0, pad_size_h))

            # Add the residual connection
            if i > 0:
                x = x + x_prev
            x_prev = x

            x = nn.Dropout(self.dropout)(x)

        # collapse the spatial dimensions of x using average pooling
        x = torch.mean(x, dim=(2, 3))

        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer
        x = self.fc(x)
        return x

class TabularResCNN(nn.Module):
    def __init__(self, act_func, dropout, channels_hidden_dim, num_classes, num_cnn_layers, kernel_size, stride,
                 use_batchnorm, pool_size):
        super(TabularResCNN, self).__init__()

        self.act_func = make_layer(act_func)
        self.dropout = dropout
        self.channels_hidden_dim = channels_hidden_dim
        self.num_cnn_layers = num_cnn_layers
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.stride = stride
        self.use_batchnorm = use_batchnorm
        self.pool_size = pool_size

        # Define the convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(num_cnn_layers):
            if i == 0:
                conv_layer = nn.Conv1d(in_channels=1, out_channels=channels_hidden_dim, kernel_size=kernel_size,
                                       stride=stride)
            else:
                conv_layer = nn.Conv1d(in_channels=channels_hidden_dim, out_channels=channels_hidden_dim, kernel_size=kernel_size,
                                       stride=stride)
            self.conv_layers.append(conv_layer)

            # Add batch normalization
            if self.use_batchnorm:
                bn_layer = make_layer('BatchNorm1d', channels_hidden_dim)
                self.conv_layers.append(bn_layer)

            # Add max-pooling layer
            pool_layer = make_layer('MaxPool1d', self.pool_size)
            self.conv_layers.append(pool_layer)

        # Define the fully connected layers
        fc_layer = nn.Linear(channels_hidden_dim, num_classes)
        self.fc = fc_layer

    def forward(self, x):
        # Save the original spatial dimensions
        original_spatial_dim = x.size(2)

        # Apply the convolutional layers
        for i, conv in enumerate(self.conv_layers):
            if i % 3 == 0:
                # Apply the convolutional layer
                x = self.act_func(conv(x))
            elif i % 3 == 1 and self.use_batchnorm:
                # Apply the batch normalization layer
                x = self.conv_layers[i](x)
            elif i % 3 == 2:
                # Apply the max-pooling layer
                x = self.conv_layers[i](x)

            # Pad the output to preserve the spatial dimensions
            pad_size = original_spatial_dim - x.size(2)
            if pad_size > 0:
                x = nn.functional.pad(x, (0, pad_size))

            # Add the residual connection
            if i > 0:
                x = x + x_prev
            x_prev = x

            x = nn.Dropout(self.dropout)(x)

        # collapse the spacial dimension of x using average pooling
        x = torch.mean(x, dim=2)
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)

        # Apply the fully connected layer
        x = self.fc(x)
        return x

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, kernel_size=80, stride=4, n_channel=128, pool_size=4):
        super(M5, self).__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size, stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(pool_size)
        self.conv2 = nn.Conv1d(n_channel, n_channel, 3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(pool_size)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, 3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(pool_size)
        self.conv4 = nn.Conv1d(2 * n_channel, 4 * n_channel, 3)
        self.bn4 = nn.BatchNorm1d(4 * n_channel)
        self.pool4 = nn.MaxPool1d(pool_size)

        self.avgPool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(4 * n_channel, n_output)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        # print(x.shape)
        x = self.avgPool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

