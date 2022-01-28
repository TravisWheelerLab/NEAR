import torch
import math
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(
            self,
            filters,
            resnet_bottleneck_factor,
            kernel_size,
            layer_index,
            first_dilated_layer,
            dilation_rate,
            stride=1,
    ):

        super(ResidualBlock, self).__init__()

        self.filters = filters

        shifted_layer_index = layer_index - first_dilated_layer + 1
        dilation_rate = int(max(1, dilation_rate ** shifted_layer_index))
        self.num_bottleneck_units = math.floor(resnet_bottleneck_factor * self.filters)
        self.bn1 = torch.nn.BatchNorm1d(self.filters)
        # need to pad 'same', so output has the same size as input
        # project down to a smaller number of self.filters with a larger kernel size
        self.conv1 = torch.nn.Conv1d(
            self.filters,
            self.num_bottleneck_units,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding="same",
        )

        self.bn2 = torch.nn.BatchNorm1d(self.num_bottleneck_units)
        # project back up to a larger number of self.filters w/ a kernel size of 1 (a local
        # linear transformation) No padding needed sin
        self.conv2 = torch.nn.Conv1d(
            self.num_bottleneck_units, self.filters, kernel_size=1, dilation=1
        )

    def _forward(self, x):
        features = self.bn1(x)
        features = torch.nn.functional.relu(features)
        features = self.conv1(features)
        features = self.bn2(features)
        features = torch.nn.functional.relu(features)
        features = self.conv2(features)
        return features + x

    def _masked_forward(self, x, mask):
        features = self.bn1(x)
        features = torch.nn.functional.relu(features)
        features = self.conv1(features)
        features[mask.expand(-1, self.num_bottleneck_units, -1)] = 0
        features = self.bn2(features)
        features = torch.nn.functional.relu(features)
        features = self.conv2(features)
        features[mask.expand(-1, self.filters, -1)] = 0
        return features + x

    def forward(self, x, mask=None):
        if mask is None:
            return self._forward(x)
        else:
            return self._masked_forward(x, mask)


