import torch
import torch.nn as nn
import torch.nn.functional as F

class ReconstructionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1):
        super(ReconstructionBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.conv_block(x)
        return x

class MultiDilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, depth = 6):
        super(MultiDilationBlock, self).__init__()

        self.convs = nn.ModuleList()
        for i in range(depth):
            dilation = 2 * (i + 1)
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        3,
                        padding=dilation,
                        dilation=dilation,
                    ),
                    nn.Conv2d(out_channels, out_channels, 1),
                )
            )

        self.conv1x1 = nn.Conv2d(depth * out_channels, out_channels, 1)

    def forward(self, x):
        outs = [conv(x) for conv in self.convs]
        out = torch.cat(outs, dim=1)
        out = self.conv1x1(out)
        return out

class ShallowFE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShallowFE, self).__init__()

        self.L1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

        self.MDB = MultiDilationBlock(out_channels, out_channels, depth = 4)

        self.SE = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(out_channels, out_channels//4, 1),
            nn.GELU(),
            nn.Conv2d(out_channels//4, out_channels, 1),
            nn.Sigmoid(),
        )

        self.L2 = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
        )

        self.skip = nn.Identity()

        self.construct = ReconstructionBlock(32, 3, 3)

    def forward(self, x):
        x = self.L1(x)

        skip1 = self.skip(x)

        x = self.MDB(x)

        skip2 = self.skip(x)

        x = self.SE(x)

        x = x * skip2

        x = torch.cat([x, skip1], dim=1)

        x = self.L2(x)

        return self.construct(x)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(self, nn.Linear):
            nn.init.kaiming_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

        elif isinstance(m, nn.Sequential):
            for layer in self:
                self.init_weights(layer)

        elif isinstance(m, MultiDilationBlock):
            for layer in m.convs:
                for l in layer:
                    self.init_weights(l)

        elif isinstance(m, ReconstructionBlock):
            for layer in m.conv_block:
                self.init_weights(layer)

        elif isinstance(m, ShallowFE):
            for layer in m.L1:
                self.init_weights(layer)

            for layer in m.L4:
                self.init_weights(layer)

            for layer in m.SE:
                self.init_weights(layer)
            
            for layer in m.MDE:
                for l in layer:
                    self.init_weights(l)
    