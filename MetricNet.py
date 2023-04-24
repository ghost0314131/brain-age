import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv3d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm3d(num_features=out_channels))
    return result


class Fork_SE_Block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', use_se=False):
        super(Fork_SE_Block, self).__init__()
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            self.se = SE_Block(out_channels, internal_neurons=out_channels // 16)
        else:
            self.se = nn.Identity()

        self.rbr_identity = nn.BatchNorm3d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
        self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)


    def forward(self, inputs):
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))


class SE_Block(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(SE_Block, self).__init__()
        self.down = nn.Conv3d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv3d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool3d(inputs, kernel_size=(inputs.size(2), inputs.size(3), inputs.size(4)))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1, 1)
        return inputs * x


class Classifier(nn.Module):

    def __init__(self, num_classes, width_multiplier, extend=False):
        super(Classifier, self).__init__()
        self.extend = extend
        if self.extend:
            self.linear_extend = nn.Linear(int(512 * width_multiplier), 256)
            self.linear = nn.Linear(256, num_classes)
        else:
            self.linear = nn.Linear(int(512 * width_multiplier), num_classes)

    def forward(self, x):
        if self.extend:
            x = self.linear_extend(x)
        f = self.linear(x)
        x = F.log_softmax(f)
        return x, f


class Fork_SE(nn.Module):

    def __init__(self, num_blocks, width_multiplier, use_se=True, override_groups_map=None, dropout=0.5):
        super(Fork_SE, self).__init__()

        assert len(width_multiplier) == 4

        self.use_se = use_se
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.in_planes = min(64, int(64 * width_multiplier[0]))

        self.stage0 = Fork_SE_Block(in_channels=1, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  use_se=self.use_se)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)  # 256
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)  # 512
        self.mid = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool3d(output_size=1)
        self.dropout = nn.Dropout(p=dropout)

    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(Fork_SE_Block(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, use_se=self.use_se))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.mid(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return x


class MetricNet(nn.Module):
    def __init__(self, num_classes=1000, num_blocks=[8, 8, 4, 1], width_multiplier=[2.5, 2.5, 2.5, 2.5], dropout=0.5):
        super(MetricNet, self).__init__()
        self.feature = Fork_SE(num_blocks=num_blocks, width_multiplier=width_multiplier, dropout=dropout)
        self.regression = Classifier(num_classes=num_classes, width_multiplier=width_multiplier[3], extend=True)

    def forward(self, x1, x2):
        m1 = self.feature(x1)
        m2 = self.feature(x2)
        o1, f1 = self.regression(m1)
        o2, f2 = self.regression(m2)
        return o1, o2, m1, m2
