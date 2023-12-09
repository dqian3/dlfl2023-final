import torch
import torch.nn as nn
from .gsta import MidMetaNet

from .simsiam import SimSiamGSTA

# Unet basic model structures.
class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some 1 by 1 conv (i.e. FF across channels)
    """

    def __init__(self, channel_seq):
        super().__init__()
        self.bridge = nn.Sequential(
            [
                ConvBlock(in_c, out_c, kernel_size=1, padding=0)
                for in_c, out_c in zip(channel_seq, channel_seq[1:])
            ]
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        elif upsampling_method == "none":
            self.upsample = ConvBlock(up_conv_in_channels, up_conv_out_channels)

        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
    
class UNetVidToSeg(nn.Module):
    def __init__(self, encoder, predictor=None, n_classes=49):
        '''
        model is trained SimSiam for Parallel2DResNet
        
        Down block outputs:
        torch.Size([1, 32, 11, 128, 128])
        torch.Size([1, 32, 11, 128, 128])
        torch.Size([1, 64, 11, 64, 64])
        torch.Size([1, 128, 11, 32, 32])
        torch.Size([1, 256, 11, 16, 16])

        '''
        super().__init__()

        down_blocks = list(encoder.children())

        bridges = [
            Bridge(c * 11, c * 5, c * 1) # 11 5 1 is frames
            for c in (32, 32, 64, 128)
        ]


        if (predictor):
            bridges.append(nn.Sequential(
                predictor,
                Bridge(256 * 11, 256 * 5, 256 * 1)
            ))
        else:
            Bridge(256 * 11, 256 * 5, 256 * 1)


        self.down_blocks = nn.ModuleList(down_blocks[:-1])
        self.bridges = nn.ModuleList(bridges)

        up_blocks = []

        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(UpBlock(256, 128))
        up_blocks.append(UpBlock(128, 64))
        up_blocks.append(UpBlock(64, 64, up_conv_in_channels=64, up_conv_out_channels=32, upsampling_method="none"))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)
        self.upsample = nn.UpsamplingBilinear2d(size=(160, 240))

    def forward(self, x):

        outputs = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)
            if i < len(self.down_blocks):
                B, C, T, H, W = x.shape
                cross_x = self.bridges[i](x.view(B, C * T, H, W))
                outputs.append(cross_x)


        for cross_x in outputs:
            print(cross_x.shape)
        print(x.shape)

        x = outputs[-1]

        for i, block in enumerate(self.up_blocks):
            # print(f"Up block {i}:")
            # print(x.shape)
            # print(outputs[-(i + 1)].shape)

            x = block(x, outputs[-(i + 1)])

        x = self.out(x)
        x = self.upsample(x)

        return x