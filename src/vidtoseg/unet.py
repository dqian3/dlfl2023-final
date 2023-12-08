import torch
import torch.nn as nn
from vidtoseg.gsta import MidMetaNet

from vidtoseg.simsiam import SimSiamGSTA

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

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
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
            self.upsample = torch.nn.Identity()

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
    def __init__(self, model: SimSiamGSTA, n_classes=49, finetune=True):
        '''
        model is trained SimSiam for r2plus1d
        
        Down block outputs:
        torch.Size([1, 32, 11, 128, 128])
        torch.Size([1, 32, 11, 128, 128])
        torch.Size([1, 64, 6, 64, 64])
        torch.Size([1, 128, 3, 32, 32])
        torch.Size([1, 256, 2, 16, 16])

        Everything except first layer is pooled to frames = 2 and vid -> channel cross connections:
        torch.Size([1, 160, 256, 256])
        torch.Size([1, 160, 128, 128])
        torch.Size([1, 64, 128, 128])
        torch.Size([1, 128, 64, 64])
        torch.Size([1, 256, 32, 32])
        torch.Size([1, 512, 16, 16])
        '''
        super().__init__()

        r2plus1d = model.backbone

        down_blocks = list(r2plus1d.children())

        cross_conns = [
            MidMetaNet(11*32, 5 * 32, 3),
            MidMetaNet(11*32, 5 * 32, 3),
            MidMetaNet(6*64, 3 * 64, 3),
            MidMetaNet(3*128, 2 * 128, 3),
            MidMetaNet(2 * 256, 256, 3),
        ]

        # Pool each down blocks ouput across time, so that we don't have too many channels
        # in the cross connection. Skip last down block, since that is the "bridge"
        down_t_pools = [
            torch.nn.AdaptiveAvgPool3d((5, None, None)),
            torch.nn.AdaptiveAvgPool3d((5, None, None)),
            torch.nn.AdaptiveAvgPool3d((2, None, None)),
            torch.nn.AdaptiveAvgPool3d((2, None, None)),
            torch.nn.AdaptiveAvgPool3d((2, None, None)),
        ]

        self.down_blocks = nn.ModuleList(down_blocks)
        self.cross_conns = nn.ModuleList(cross_conns)
        self.down_t_pools = nn.ModuleList(down_t_pools)

        self.bridge = model.predictor

        up_blocks = []

        up_blocks.append(UpBlock(512, 256))
        up_blocks.append(UpBlock(256, 128))
        up_blocks.append(UpBlock(128 + 160, 128, up_conv_in_channels=128, up_conv_out_channels=128))
        up_blocks.append(UpBlock(128 + 160, 128, up_conv_in_channels=128, up_conv_out_channels=128, upsampling_method="none"))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out = nn.Conv2d(128, n_classes, kernel_size=1, stride=1)

        self.upsample = nn.UpsamplingBilinear2d(size=(160, 240))

    def forward(self, x):
        # TODO adjust finetuning behavior

        outputs = []
        for i, block in enumerate(self.down_blocks):
            x = block(x)

            # print(len(self.down_blocks))
            # print(i)

            if i < len(self.down_blocks) - 1:
                # avg pool the cross connections
                cross_x = self.cross_conns[i](x)
                cross_x = self.down_t_pools[i](cross_x)
                B, C, T, H, W = cross_x.shape # pool cross sections
                cross_x = cross_x.view(B, C * T, H, W)
                outputs.append(cross_x)

        # for cross_x in outputs:
        #     print(cross_x.shape)
        # print(x.shape)

        x = self.bridge(x)
        B, C, T, H, W = x.shape
        x = x.view(B, C * T, H, W)


        for i, block in enumerate(self.up_blocks):
            # print(f"Up block {i}:")
            # print(x.shape)
            # print(outputs[-(i + 1)].shape)

            x = block(x, outputs[-(i + 1)])

        x = self.out(x)
        x = self.upsample(x)

        return x