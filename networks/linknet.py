"""
Codes of LinkNet based on https://github.com/snakers4/spacenet-three
"""
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from .EAEFNet import Block
from networks.common import nonlinearity, DecoderBlock


class LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.firstconv_t = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        ratio = 1
        c1 = int(64 * ratio)
        c2 = int(64 * ratio)
        c3 = int(128 * ratio)
        c4 = int(256 * ratio)    
        c5 = int(512 * ratio)    
        #self.block1 = Block([c1, c1, 'M'], in_channels=c1, first_block=True)
        #self.block2 = Block([c2, c2, 'M'], in_channels=c1)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
        self.block5 = Block([c5, c5, c5, c5, 'M'], in_channels=c4)    
        self.block6 = Block([c5, c5, c5, c5], in_channels=c5)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, x):
        if 1:
            # Encoder
            x = self.firstconv(x)
            x = self.firstbn(x)
            x = self.firstrelu(x)
            x = self.firstmaxpool(x)#[16, 64, 128, 128])
            e1 = self.encoder1(x)#[16, 64, 128, 128]
            e2 = self.encoder2(e1)#[16, 128, 64, 64]
            e3 = self.encoder3(e2)#[16, 256, 32, 32]
            e4 = self.encoder4(e3)#[16, 512, 16, 16]
        else:

            RGB = self.firstconv(RGB)#[1, 64, 256, 256])
            RGB = self.firstbn(RGB)
            RGB = self.firstrelu(RGB)

            T = self.firstconv_t(T)#[1, 64, 256, 256])
            T = self.firstbn(T)
            T = self.firstrelu(T)

            # RGB = x
            # T = x

            #RGB, T, e0 = self.block1(RGB, T)#
            RGB, T, e1 = self.block2(RGB, T)#[1, 128, 128, 128])
            RGB, T, e2 = self.block3(RGB, T) #[1, 256, 64, 64])
            RGB, T, e3 = self.block4(RGB, T) #[1, 512, 32, 32])
            RGB, T, e4 = self.block5(RGB, T) #[1, 512, 32, 32])
            RGB, T, e4 = self.block6(RGB, T) #[1, 512, 32, 32])
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)
class rgbm_LinkNet34(nn.Module):
    def __init__(self, num_classes=1):
        super(rgbm_LinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=False)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.firstconv_t = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        ratio = 1
        c1 = int(64 * ratio)
        c2 = int(64 * ratio)
        c3 = int(128 * ratio)
        c4 = int(256 * ratio)    
        c5 = int(512 * ratio)    
        #self.block1 = Block([c1, c1, 'M'], in_channels=c1, first_block=True)
        #self.block2 = Block([c2, c2, 'M'], in_channels=c1)
        self.block2 = Block([c2, c2, 'M'], in_channels=c1)
        self.block3 = Block([c3, c3, c3, c3, 'M'], in_channels=c2)
        self.block4 = Block([c4, c4, c4, c4, 'M'], in_channels=c3)
        self.block5 = Block([c5, c5, c5, c5, 'M'], in_channels=c4)    
        self.block6 = Block([c5, c5, c5, c5], in_channels=c5)


        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 2, padding=1)

    def forward(self, RGB,T):
        if 0:
            # Encoder
            x = self.firstconv(x)
            x = self.firstbn(x)
            x = self.firstrelu(x)
            x = self.firstmaxpool(x)#[16, 64, 128, 128])
            e1 = self.encoder1(x)#[16, 64, 128, 128]
            e2 = self.encoder2(e1)#[16, 128, 64, 64]
            e3 = self.encoder3(e2)#[16, 256, 32, 32]
            e4 = self.encoder4(e3)#[16, 512, 16, 16]
        else:

            RGB = self.firstconv(RGB)#[1, 64, 256, 256])
            RGB = self.firstbn(RGB)
            RGB = self.firstrelu(RGB)

            T = self.firstconv_t(T)#[1, 64, 256, 256])
            T = self.firstbn(T)
            T = self.firstrelu(T)

            # RGB = x
            # T = x

            #RGB, T, e0 = self.block1(RGB, T)#
            RGB, T, e1 = self.block2(RGB, T)#[1, 128, 128, 128])
            RGB, T, e2 = self.block3(RGB, T) #[1, 256, 64, 64])
            RGB, T, e3 = self.block4(RGB, T) #[1, 512, 32, 32])
            RGB, T, _ = self.block5(RGB, T) #[1, 512, 32, 32])
            RGB, T, e4 = self.block6(RGB, T) #[1, 512, 32, 32])
        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

