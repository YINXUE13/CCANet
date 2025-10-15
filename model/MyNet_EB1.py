# coding=utf-8
from .basic import *
from .EfficientNet import EfficientNet_B0
from thop import profile


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        # Backbone model
        self.backbone = EfficientNet_B0()
        self.GCM = GCM(32, 32)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear')
        self.upsample5 = nn.Upsample(scale_factor=32, mode='bilinear')

        self.reduce1 = DepthwiseSeparableConv(16, 32)
        self.reduce2 = DepthwiseSeparableConv(24, 32)
        self.reduce3 = DepthwiseSeparableConv(40, 32)
        self.reduce4 = DepthwiseSeparableConv(112, 32)
        self.reduce5 = DepthwiseSeparableConv(320, 32)

        self.FIM_1 = HFIM(32, 32)
        self.FIM_2 = HFIM(32, 32)
        self.FIM_3 = HFIM(32, 32)

        self.ADB_1 = ADB(64,32)
        self.ADB_2 = ADB(64,32)
        self.ADB_3 = ADB(64,32)
        self.ADB_4 = ADB(64,32)
        self.ADB_5 = ADB(32,32)

        self.conv_out1 = nn.Conv2d(32, 1, 3, padding=1)
        self.conv_out2 = nn.Conv2d(32, 1, 3, padding=1)
        self.conv_out3 = nn.Conv2d(32, 1, 3, padding=1)
        self.conv_out4 = nn.Conv2d(32, 1, 3, padding=1)
        self.conv_out5 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        # generate backbone features
        ff1, ff2, ff3, ff4, ff5 = self.backbone(x)

        f1 = self.reduce1(ff1)
        f2 = self.reduce2(ff2)
        f3 = self.reduce3(ff3)
        f4 = self.reduce4(ff4)
        f5 = self.reduce5(ff5)

        f5 = self.GCM(f5)
        p1 = self.FIM_1(f2, f5, f1)
        p2 = self.FIM_2(f3, f5, f1)
        p3 = self.FIM_3(f4, f5, f1)

        a5 = self.ADB_5(f5)
        c4 = torch.cat([self.upsample1(a5), p3], 1)
        a4 = self.ADB_4(c4)
        c3 = torch.cat([self.upsample1(a4), p2], 1)
        a3 = self.ADB_3(c3)
        c2 = torch.cat([self.upsample1(a3), p1], 1)
        a2 = self.ADB_2(c2)
        c1 = torch.cat([self.upsample1(a2), f1], 1)
        a1 = self.ADB_1(c1)

        s1 = self.upsample1(self.conv_out1(a1))
        s2 = self.upsample2(self.conv_out2(a2))
        s3 = self.upsample3(self.conv_out3(a3))
        s4 = self.upsample4(self.conv_out4(a4))
        s5 = self.upsample5(self.conv_out5(a5))

        ff5 = self.upsample5(ff5)
        # print('s1.shape', s1.shape)
        # print('s2.shape', s2.shape)
        # print('s3.shape', s3.shape)
        # print('s4.shape', s4.shape)
        # print('s5.shape', s5.shape)

        return s1, torch.sigmoid(s1), s2, torch.sigmoid(s2), s3, torch.sigmoid(s3), s4, torch.sigmoid(s4), s5, torch.sigmoid(s5),ff5


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNet().to(device)
    dummy_input = torch.randn(1, 3, 352, 352)
    dummy_input = dummy_input.to(device)
    flops, params = profile(model, (dummy_input,))
    print('flops: ', flops, 'params: ', params)
    print('flops: %.2f G, params: %.2f M' % (flops / 1e9, params / 1e6))

