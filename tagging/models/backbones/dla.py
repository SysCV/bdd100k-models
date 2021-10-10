"""DLA Backbone."""
# mypy: ignore-errors

import math

import torch
from mmcls.models.builder import BACKBONES
from torch import nn

BatchNorm = nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def make_conv_level(
    inplanes, planes, convs, stride=1, dilation=1
) -> nn.Sequential:
    """Make conv block."""
    modules = []
    for i in range(convs):
        modules.extend(
            [
                nn.Conv2d(
                    inplanes,
                    planes,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    padding=dilation,
                    bias=False,
                    dilation=dilation,
                ),
                BatchNorm(planes),
                nn.ReLU(inplace=True),
            ]
        )
        inplanes = planes
    return nn.Sequential(*modules)


class BasicBlock(nn.Module):  # type: ignore
    """Basic block."""

    def __init__(self, inplanes, planes, stride=1, dilation=1) -> None:
        """Init function."""
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, input_x, residual=None):
        """Forward pass."""
        if residual is None:
            residual = input_x

        out = self.conv1(input_x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):  # type: ignore
    """Bottleneck."""

    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1) -> None:
        """Init function."""
        super().__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False
        )
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
        )
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, input_x, residual=None):
        """Forward pass."""
        if residual is None:
            residual = input_x

        out = self.conv1(input_x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class BottleneckX(nn.Module):  # type: ignore
    """BottleneckX."""

    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1) -> None:
        """Init function."""
        super().__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(
            inplanes, bottle_planes, kernel_size=1, bias=False
        )
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(
            bottle_planes,
            bottle_planes,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            bias=False,
            dilation=dilation,
            groups=cardinality,
        )
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(
            bottle_planes, planes, kernel_size=1, bias=False
        )
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, input_x, residual=None):
        """Forward pass."""
        if residual is None:
            residual = input_x

        out = self.conv1(input_x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):  # type: ignore
    """Root."""

    def __init__(
        self, in_channels, out_channels, kernel_size, residual
    ) -> None:
        """Init function."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            bias=False,
            padding=(kernel_size - 1) // 2,
        )
        self.bn = BatchNorm(out_channels)  # pylint: disable=invalid-name
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *input_x):
        """Forward pass."""
        children = input_x
        input_x = self.conv(torch.cat(input_x, 1))
        input_x = self.bn(input_x)
        if self.residual:
            input_x += children[0]
        input_x = self.relu(input_x)

        return input_x


class Tree(nn.Module):  # type: ignore
    """Tree."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        levels,
        block,
        in_channels,
        out_channels,
        stride=1,
        level_root=False,
        root_dim=0,
        root_kernel_size=1,
        dilation=1,
        root_residual=False,
    ) -> None:
        """Init function."""
        super().__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(
                in_channels, out_channels, stride, dilation=dilation
            )
            self.tree2 = block(
                out_channels, out_channels, 1, dilation=dilation
            )
        else:
            self.tree1 = Tree(
                levels - 1,
                block,
                in_channels,
                out_channels,
                stride,
                root_dim=0,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
            self.tree2 = Tree(
                levels - 1,
                block,
                out_channels,
                out_channels,
                root_dim=root_dim + out_channels,
                root_kernel_size=root_kernel_size,
                dilation=dilation,
                root_residual=root_residual,
            )
        if levels == 1:
            self.root = Root(
                root_dim, out_channels, root_kernel_size, root_residual
            )
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                BatchNorm(out_channels),
            )

    def forward(self, input_x, residual=None, children=None):
        """Forward pass."""
        children = [] if children is None else children
        bottom = self.downsample(input_x) if self.downsample else input_x
        residual = self.project(bottom) if self.project else bottom
        if self.level_root:
            children.append(bottom)
        input_x1 = self.tree1(input_x, residual)
        if self.levels == 1:
            input_x2 = self.tree2(input_x1)
            input_x = self.root(input_x2, input_x1, *children)
        else:
            children.append(input_x1)
            input_x = self.tree2(input_x1, children=children)
        return input_x


@BACKBONES.register_module()
class DLA(nn.Module):  # type: ignore
    """DLA Backbone."""

    # pylint: disable=too-many-instance-attributes

    def __init__(
        self,
        levels,
        channels,
        num_classes=1000,
        block="BasicBlock",
        residual_root=False,
        return_levels=False,
        pool_size=7,
        linear_root=False,
        norm_eval=False,
    ):
        """Init function."""
        super().__init__()
        assert block in ["BasicBlock", "Bottleneck", "BottleneckX"]
        if block == "BasicBlock":
            block = BasicBlock
        elif block == "Bottleneck":
            block = Bottleneck
        elif block == "BottleneckX":
            block = BottleneckX

        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.linear_root = linear_root
        self.norm_eval = norm_eval
        self.base_layer = nn.Sequential(
            nn.Conv2d(
                3, channels[0], kernel_size=7, stride=1, padding=3, bias=False
            ),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.level0 = make_conv_level(channels[0], channels[0], levels[0])
        self.level1 = make_conv_level(
            channels[0], channels[1], levels[1], stride=2
        )
        self.level2 = Tree(
            levels[2],
            block,
            channels[1],
            channels[2],
            2,
            level_root=False,
            root_residual=residual_root,
        )
        self.level3 = Tree(
            levels[3],
            block,
            channels[2],
            channels[3],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level4 = Tree(
            levels[4],
            block,
            channels[3],
            channels[4],
            2,
            level_root=True,
            root_residual=residual_root,
        )
        self.level5 = Tree(
            levels[5],
            block,
            channels[4],
            channels[5],
            2,
            level_root=True,
            root_residual=residual_root,
        )

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(  # pylint: disable=invalid-name
            channels[-1],
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, input_x):
        """Forward pass."""
        input_x = self.base_layer(input_x)
        for i in range(6):
            input_x = getattr(self, f"level{i}")(input_x)
        input_x = self.avgpool(input_x)
        input_x = self.fc(input_x)
        input_x = input_x.view(input_x.size(0), -1)

        return input_x

    def train(self, mode=True) -> None:
        """Set train mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, BatchNorm):
                    m.eval()
