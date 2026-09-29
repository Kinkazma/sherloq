# Local SHERLOQ adaptation; original: models/external/clone_detectors/03_cmsegnet/predict_folder/model.py
# Source SHA256: cb91e5b08f91a33fa7714e1d39c07902e103e4114be6833802017c63dbcafed5
"""
Created on Mon Feb 19 11:09:45 2024

@author: nchuvsp
"""


import torch.nn as nn


import math


import os


import torchvision.models as models


import torch.nn.functional as F


import torch


from .Cor import Corr


class MobileNetV2(nn.Module):

    """
    from MobileNetV2 import MobileNetV2

    net = MobileNetV2(n_class=1000)
    state_dict = torch.load('mobilenetv2.pth.tar') # add map_location='cpu' if no gpu
    net.load_state_dict(state_dict)
    """

    def __init__(self, n_class=1000, input_size=512, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        
        #x = x.mean(3).mean(2)
        #x = self.classifier(x)
        
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


class SpatialAttention(nn.Module):
    def __init__(self, ):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        input = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        y = torch.sigmoid(x)
        return input + input * y        


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class UnetMobilenetV2(nn.Module):
    def __init__(self, num_classes=1, num_filters=32, pretrained=True,
                 Dropout=.2, path=os.path.join(r'mobilenet_v2.pth.tar')):
        super(UnetMobilenetV2, self).__init__()
        self.encoder = MobileNetV2(n_class=1000)

        self.num_classes = num_classes

        self.dconv1 = nn.ConvTranspose2d(1280, 96, 4, padding=1, stride=2)
        self.invres1 = InvertedResidual(192, 96, 1, 6)

        self.dconv2 = nn.ConvTranspose2d(96, 32, 4, padding=1, stride=2)
        self.invres2 = InvertedResidual(64, 32, 1, 6)

        self.dconv3 = nn.ConvTranspose2d(32, 24, 4, padding=1, stride=2)
        self.invres3 = InvertedResidual(48, 24, 1, 6)

        self.dconv4 = nn.ConvTranspose2d(24, 16, 4, stride=4, padding=0)
        self.invres4 = InvertedResidual(32, 16, 1, 6)
        
        self.dconv5 = nn.ConvTranspose2d(16, 3, 4, padding=1, stride=2)
        self.invres5 = InvertedResidual(6, 3, 1, 6)
        
        self.trans = nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        
        self.conv_last = nn.Conv2d(16, 3, 1)

        self.conv_score = nn.Conv2d(3, 1, 1)

        #doesn't needed; obly for compatibility
        self.dconv_final = nn.ConvTranspose2d(1, 1, 4, padding=1, stride=2)

        if pretrained:
            state_dict = torch.load(path)
            self.encoder.load_state_dict(state_dict)
        else: pass  # Full checkpoint loaded strictly by the adapter.

        #############################

        self.corr16 = Corr(topk=16)

        self.corr24 = Corr(topk=24)

        self.corr32 = Corr(topk=32)

        self.corr96 = Corr(topk=96)
        
        
        self.aspp1 = models.segmentation.deeplabv3.ASPP(in_channels=96, out_channels=96,atrous_rates=[4, 8, 12,16])
        self.aspp2 = models.segmentation.deeplabv3.ASPP(in_channels=32,out_channels=32, atrous_rates=[4, 8, 12,16])
        self.aspp3 = models.segmentation.deeplabv3.ASPP(in_channels=24,out_channels=24, atrous_rates=[4, 8, 12,16])
        self.aspp4 = models.segmentation.deeplabv3.ASPP(in_channels=16,out_channels=16, atrous_rates=[4, 8, 12,16])

        self.sam1 = SpatialAttention()
        self.sam2 = SpatialAttention()
        self.sam3 = SpatialAttention()
        self.sam4 = SpatialAttention()
        #self.sam5 = SpatialAttention()
        #############################

    def forward(self, x):
        #print('x:',x.shape)
        for n in range(0, 2):
            x = self.encoder.features[n](x)
        x1 = x
        #x1 = self.corr16(x1)
        #print("x1",x1.shape)
        x1 = self.aspp4(x1)
        #print("x1_aspp4",x1.shape)
        x1 = self.sam1(x1)
        
        #print("x1_sam1",x1.shape)
        #print('x1:',x.shape)

        for n in range(2, 4):
            x = self.encoder.features[n](x)
        x2 = x
        #print("x2",x2.shape)
        x2 = self.corr24(x2)
        #print("x2_corrr",x2.shape)
        x2 = self.aspp3(x2)
        #print("x2_aspp3",x2.shape)
        x2 = self.sam2(x2)
        
        #print("x2_sam2",x2.shape)
        #print('x2:',x.shape)

        for n in range(4, 7):
            x = self.encoder.features[n](x)
        x3 = x
        #print("x3",x3.shape)
        x3 = self.corr32(x3)
        #print("x3_corrr",x3.shape)
        x3 = self.aspp2(x3)
        #print("x3_aspp2",x3.shape)
        x3 = self.sam3(x3)
        
        #print("x3_sam3",x3.shape)
        #print('x3:',x.shape)

        for n in range(7, 14):
            x = self.encoder.features[n](x)
        x4 = x
        #print("x4",x4.shape)
        x4 = self.corr96(x4)
        #print("x4_corrr",x4.shape)
        x4 = self.aspp1(x4)
        #print("x4_aspp1",x4.shape)
        x4 = self.sam4(x4)
        
        #print("x4_sam4",x4.shape)

        for n in range(14, 19):
            x = self.encoder.features[n](x)
        x5=x
        #x5= self.corr96(x5)
        #print('x5:',x.shape)

        up1 = torch.cat([
            x4,
            self.dconv1(x5)
        ], dim=1)
        #print('up1:',up1.shape)
        up1 = self.invres1(up1)
        #print('up1_invres:',up1.shape)


        up2 = torch.cat([
            x3,
            self.dconv2(up1)
        ], dim=1)
        #print('up2:',up2.shape)
        up2 = self.invres2(up2)
        #print('up2_invres:',up2.shape)

        up3 = torch.cat([
            x2,
            self.dconv3(up2)
        ], dim=1)
        #print('up3:',up3.shape)
        up3 = self.invres3(up3) 
        #print('up3_invres:',up3.shape)

        up4 = torch.cat([
            self.trans(x1),
            self.dconv4(up3)
        ], dim=1)
        #print('up4:',up4.shape)
        up4 = self.invres4(up4)
        #print('up4_invres:',up4.shape)
        
        x = self.conv_last(up4)
        #print('x_last',x.shape)
        x = self.conv_score(x)
        #print('x_score',x.shape)

        return x
    """
from torchsummary import summary

# 假设您的模型是UnetMobilenetV2
model = UnetMobilenetV2()  # 假设不使用预训练权重

# 设置模型为评估模式
model.eval()

# 使用summary函数
# 您需要根据您模型的输入尺寸来调整这里的(3, H, W)，这里假设输入尺寸是(3, 512, 512)
summary(model, input_size=(3, 256, 384), device='cpu')
"""


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.drop4 = nn.Dropout(0.5)
        self.pool4 = nn.MaxPool2d(2)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.drop5 = nn.Dropout(0.5)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.conv10 = nn.Conv2d(2, 1, 1)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.conv4(pool3)
        drop4 = self.drop4(conv4)
        pool4 = self.pool4(drop4)
        
        conv5 = self.conv5(pool4)
        drop5 = self.drop5(conv5)
        
        up6 = self.up6(drop5)
        merge6 = torch.cat([drop4, up6], dim=1)
        conv6 = self.conv6(merge6)
        
        up7 = self.up7(conv6)
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = self.conv7(merge7)
        
        up8 = self.up8(conv7)
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = self.conv8(merge8)
        
        up9 = self.up9(conv8)
        merge9 = torch.cat([conv1, up9], dim=1)
        conv9 = self.conv9(merge9)
        
        conv10 = self.conv10(conv9)
        
        return conv10


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.match_channels = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.match_channels:
            residual = self.match_channels(residual)
        out += residual
        out = F.relu(out)
        return out


class RecurrentBlock(nn.Module):
    def __init__(self, out_channels, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        for _ in range(self.t):
            x = F.relu(self.bn(self.conv(x)))
        return x


class RRUBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t=2):
        super(RRUBlock, self).__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels)
        self.recurrent_block = RecurrentBlock(out_channels, t)

    def forward(self, x):
        out = self.residual_block(x)
        out = self.recurrent_block(out)
        return out


class RRUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RRUNet, self).__init__()
        self.rru1 = RRUBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.rru2 = RRUBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.rru3 = RRUBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.rru4 = RRUBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        self.rru5 = RRUBlock(512, 1024)
        
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.rru6 = RRUBlock(1024, 512)
        
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.rru7 = RRUBlock(512, 256)
        
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.rru8 = RRUBlock(256, 128)
        
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.rru9 = RRUBlock(128, 64)
        
        self.conv10 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.rru1(x)
        p1 = self.pool1(x1)
        
        x2 = self.rru2(p1)
        p2 = self.pool2(x2)
        
        x3 = self.rru3(p2)
        p3 = self.pool3(x3)
        
        x4 = self.rru4(p3)
        p4 = self.pool4(x4)
        
        x5 = self.rru5(p4)
        
        up6 = self.up6(x5)
        merge6 = torch.cat([up6, x4], dim=1)
        x6 = self.rru6(merge6)
        
        up7 = self.up7(x6)
        merge7 = torch.cat([up7, x3], dim=1)
        x7 = self.rru7(merge7)
        
        up8 = self.up8(x7)
        merge8 = torch.cat([up8, x2], dim=1)
        x8 = self.rru8(merge8)
        
        up9 = self.up9(x8)
        merge9 = torch.cat([up9, x1], dim=1)
        x9 = self.rru9(merge9)
        
        out = self.conv10(x9)
        return out


"""
# Initialize the modified model
model = UnetMobilenetV2()

# Generate a random input tensor with shape (1, 3, 256, 384)
input_tensor = torch.randn(2, 3, 256, 384)

# Perform a forward pass through the model
output = model(input_tensor)

# Print the shape of the output tensor
output_shape = output.shape
output_shape
"""

