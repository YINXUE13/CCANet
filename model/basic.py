import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange
import math

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, bias=True, se_ratio=2):
        super(DepthwiseSeparableConv, self).__init__()

        if dilation != 1:
            padding = dilation

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                                   groups=in_channels, bias=False, stride=stride)
        self.bn = nn.BatchNorm2d(in_channels)
        self.se = SqueezeExcitation(in_channels=in_channels, out_channels=in_channels, reduction=se_ratio)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = torch.relu(self.bn(out))
        out = self.se(out)
        out = self.pointwise(out)
        return out

class SqueezeExcitation(nn.Module):
    """
    Modified from the official implementation of [torchvision.ops.SqueezeExcitation]
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): reduction ratio
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            reduction: int = 4,
            activation=nn.ReLU,
            scale_activation=nn.Sigmoid,
            pool='avgpool'
    ):
        super(SqueezeExcitation, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels != out_channels:
            self.transition = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )

        if out_channels // reduction == 0:
            reduction = 1

        if pool == 'avgpool':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pool == 'maxpool':
            self.pool = nn.AdaptiveMaxPool2d(1)
        else:
            print('Parameter pool is not avgpool or maxpool')
            return
        self.fc1 = nn.Conv2d(out_channels, out_channels // reduction, 1)
        self.fc2 = nn.Conv2d(out_channels // reduction, out_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, x):
        scale = self.pool(x)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            x = self.transition(x)
        scale = self._scale(x)
        return scale * x




class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        return self.conv(x)



def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # channel shuffle, 通道洗牌
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class ADB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ADB, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)

        self.conv1 = convbnrelu(out_channel, out_channel, k=3, p=1, g=out_channel)
        self.conv2 = convbnrelu(out_channel, out_channel, k=(1, 3), p=(0, 1), g=out_channel)
        self.conv3 = convbnrelu(out_channel, out_channel, k=(3, 1), p=(1, 0), g=out_channel)

        self.relu = nn.ReLU(inplace=True)

        # 加权空间注意力模块
        self.weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.sa_fusion = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.SA = nn.ModuleList([SA() for _ in range(3)])  # 包含每一份的 SA 模块

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x0)
        x3 = self.conv3(x0)

        # 将 x1, x2, x3 串联后经过通道混洗
        x = torch.cat([x1, x2, x3], dim=1)
        x = channel_shuffle(x, 3)

        # 将混洗后的 x 分成 3 份
        x_splits = torch.chunk(x, 3, dim=1)

        # 对每份应用空间注意力
        sa_outputs = [self.SA[i](x_splits[i]) for i in range(3)]

        # 计算加权和
        nor_weights = F.softmax(self.weight, dim=0)
        weighted_sum = sum(sa_outputs[i] * nor_weights[i] for i in range(3))

        # 将加权和传入 sa_fusion
        attention_out = self.sa_fusion(weighted_sum)

        # 残差连接输出
        out = self.relu(attention_out * x0 + x0)

        return out


###########  ASM  ############
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class ASM(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False):
        super(ASM, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.conv1x1_1 = nn.Conv2d(dim, dim//2, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(dim, dim//2, kernel_size=1)
        self.norm1 = LayerNorm(dim//2)
        self.norm2 = LayerNorm(dim//2)

        self.kv = nn.Conv2d(dim//2, dim, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.conv1x1 = nn.Conv2d(dim, dim//2, kernel_size=1)

        self.q = nn.Conv2d(dim//2, dim//2, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim//2, dim//2, kernel_size=3, stride=1, padding=1, groups=dim//2, bias=bias)

        self.project_out = nn.Conv2d(dim//2, dim, kernel_size=1, bias=bias)
        self.conv_cat = nn.Conv2d(dim*2, dim, kernel_size=1)

    def forward(self, x, y):
        x1 = self.conv1x1_1(x)
        y1 = self.conv1x1_2(y)
        x1 = self.norm1(x1)
        y1 = self.norm2(y1)

        b, c, h, w = x1.shape
        kv = self.kv_dwconv(self.kv(x1))
        k, v = kv.chunk(2, dim=1)

        k = F.relu(k)
        v = -1*torch.min(v.float(), torch.tensor(0).float())
        k = self.conv1x1(torch.cat([k, v], dim=1))

        q = self.q_dwconv(self.q(y1))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out) + x

        out = self.conv_cat(torch.cat([y, out], dim=1))
        return out



class HFIM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HFIM, self).__init__()
        self.conv1 = convbnrelu(in_channel * 3, out_channel, 3)
        self.tru = ASM(out_channel)
        self.mam = FIFM(out_channel)

    def forward(self, x, y, z):
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        z = F.interpolate(z, size=x.shape[2:], mode='bilinear', align_corners=False)

        x = self.conv1(torch.cat([x, y, z], dim=1))
        xg = self.tru(x, y)
        xm = self.mam(xg, z)
        out = xg * xm + x
        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(GCM, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x



class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class FIFM(nn.Module):
    def __init__(self, channel):
        super(FIFM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel// 2, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(channel // 2, channel // 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU()
        )

        self.conv3 = nn.Sequential(
            DepthwiseSeparableConv(channel, channel // 4, kernel_size=(1, 3), padding=(0, 1)),
            DepthwiseSeparableConv(channel // 4, channel // 4, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.conv4 = nn.Sequential(
            DepthwiseSeparableConv(channel, channel // 4, kernel_size=(1, 5), padding=(0, 2)),
            DepthwiseSeparableConv(channel // 4, channel // 4, kernel_size=(5, 1), padding=(2, 0)),
        )
        self.conv5 = nn.Sequential(
            DepthwiseSeparableConv(channel, channel//4, kernel_size=(1, 7), padding=(0, 3)),
            DepthwiseSeparableConv(channel // 4, channel // 4, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.gap1 = _AsppPooling(channel, channel//4)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )

        self.conv6 = nn.Sequential(
            DepthwiseSeparableConv(channel, channel // 4, kernel_size=(1, 3), padding=(0, 1)),
            DepthwiseSeparableConv(channel // 4, channel // 4, kernel_size=(3, 1), padding=(1, 0)),
        )
        self.conv7 = nn.Sequential(
            DepthwiseSeparableConv(channel, channel // 4, kernel_size=(1, 5), padding=(0, 2)),
            DepthwiseSeparableConv(channel // 4, channel // 4, kernel_size=(5, 1), padding=(2, 0)),
        )
        self.conv8 = nn.Sequential(
            DepthwiseSeparableConv(channel, channel//4, kernel_size=(1, 7), padding=(0, 3)),
            DepthwiseSeparableConv(channel // 4, channel // 4, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.gap2 = _AsppPooling(channel, channel // 4)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channel),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1)

    def forward(self, x, y):
        x, xx = torch.split(x, [16, 16], dim=1)
        y, yy = torch.split(y, [16, 16], dim=1)
        x_1 = self.conv1(x)
        y_1 = self.conv2(y)

        xy = torch.cat([x_1, y], dim=1)
        yx = torch.cat([y_1, x], dim=1)
        z = torch.cat((xx, yy), dim=1)

        x_w = torch.cat([self.conv3(z), self.conv4(z),self.conv5(z), self.gap1(z)], dim=1)
        x_w = self.fuse1(x_w).unsqueeze(1)

        y_w = torch.cat((self.conv6(z), self.conv7(z),self.conv8(z), self.gap2(z)), dim=1)
        y_w = self.fuse2(y_w).unsqueeze(1)

        weights = self.softmax(torch.cat((x_w, y_w), dim=1))
        x_w, y_w = weights[:, 0:1, :, :, :].squeeze(1), weights[:, 1:2, :, :, :].squeeze(1)

        out = xy.mul(x_w) + yx.mul(y_w)
        out = self.conv1x1(out)

        return out
