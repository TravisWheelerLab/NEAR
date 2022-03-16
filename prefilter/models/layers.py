import torch
import math
import torch.nn as nn

__all__ = ["ResidualBlock"]


class MultiReceptiveFieldBlock(nn.Module):
    def __init__(
        self,
        in_filters,
        out_filters,
        min_stride=10,
        max_stride=50,
        min_dilation=1,
        max_dilation=10,
        min_kernel_size=3,
        max_kernel_size=11,
    ):

        super(MultiReceptiveFieldBlock, self).__init__()

        self.in_filters = in_filters
        self.out_filters = out_filters
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.min_dilation = min_dilation
        self.max_dilation = max_dilation
        self.min_kernel_size = min_kernel_size
        self.max_kernel_size = max_kernel_size

        self.dilations = range(self.min_dilation, self.max_dilation, 2)
        self.kernels = range(self.min_kernel_size, self.max_kernel_size + 2, 2)
        self.act = torch.nn.ReLU()
        self.layers = []
        self._setup_layers()

        for i, l in enumerate(self.layers):
            setattr(self, f"conv{i}", l)

    def _setup_layers(self):

        for i, stride in enumerate(range(self.min_stride, self.max_stride, 10)[::-1]):
            # smaller kernel, larger stride, smaller dilation
            self.layers.append(
                torch.nn.Conv1d(
                    self.in_filters,
                    self.out_filters,
                    kernel_size=self.kernels[i],
                    dilation=self.dilations[i],
                    stride=stride,
                )
            )

    def _forward(self, x):
        acts = []
        for conv in self.layers:
            acts.append(self.act(conv(x)))
        return acts

    def _masked_forward(self, x, mask):
        acts = []
        for conv in self.layers:
            y = self.act(conv(x))
            y[mask.expand(-1, self.out_filters, -1)] = 0
            acts.append(y)
        return acts

    def forward(self, x, mask=None):
        if mask is None:
            return torch.cat(self._forward(x), dim=-1)
        else:
            return torch.cat(self._masked_forward(x, mask), dim=-1)


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
        typ="1d",
    ):

        super(ResidualBlock, self).__init__()

        self.filters = filters

        shifted_layer_index = layer_index - first_dilated_layer + 1
        if dilation_rate is not None:
            dilation_rate = int(max(1, dilation_rate ** shifted_layer_index))
        else:
            dilation_rate = 1
        self.num_bottleneck_units = math.floor(resnet_bottleneck_factor * self.filters)
        if typ == "1d":
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
        else:
            self.bn1 = torch.nn.BatchNorm2d(self.filters)
            # need to pad 'same', so output has the same size as input
            # project down to a smaller number of self.filters with a larger kernel size
            self.conv1 = torch.nn.Conv2d(
                self.filters,
                self.num_bottleneck_units,
                kernel_size=kernel_size,
                dilation=dilation_rate,
                padding="same",
            )

            self.bn2 = torch.nn.BatchNorm2d(self.num_bottleneck_units)
            # project back up to a larger number of self.filters w/ a kernel size of 1 (a local
            # linear transformation) No padding needed sin
            self.conv2 = torch.nn.Conv2d(
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
