import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FuseModule(nn.Module):
    def __init__(self, in_dim=64):
        super(FuseModule, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)
        self.key_conv = nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=True)

        self.gamma1 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.gamma2 = nn.Conv2d(in_dim * 2, 2, 3, 1, 1, bias=True)
        self.sig = nn.Sigmoid()

        self.se_x = SELayer(in_dim)
        self.se_prior = SELayer(in_dim)

    def forward(self, x, prior):
        x = self.se_x(x)
        prior = self.se_prior(prior)

        x_q = self.query_conv(x)
        prior_k = self.key_conv(prior)
        energy = x_q * prior_k
        attention = self.sig(energy)
        attention_x = x * attention
        attention_p = prior * attention

        x_gamma = self.gamma1(torch.cat((x, attention_x), dim=1))
        x_out = x * x_gamma[:, [0], :, :] + attention_x * x_gamma[:, [1], :, :]

        p_gamma = self.gamma2(torch.cat((prior, attention_p), dim=1))
        prior_out = prior * p_gamma[:, [0], :, :] + attention_p * p_gamma[:, [1], :, :]

        return x_out, prior_out


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # LH, 90 degree
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        # HL, 0 degree
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        # HH, 45 / 135 degree
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x):
        if x.shape[2] % 2 == 1:
            x = F.pad(x, [0, 0, 0, 1])
        if x.shape[3] % 2 == 1:
            x = F.pad(x, [0, 1, 0, 0])

        out = F.conv2d(x, self.haar_weights.to(x.device).to(x.dtype), bias=None,
                       stride=2, groups=self.channel_in) / 4.0

        out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
        return out


class InvHaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(InvHaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)
        self.haar_weights[1, 0, 0, 1] = -1  # 90
        self.haar_weights[1, 0, 1, 1] = -1
        self.haar_weights[2, 0, 1, 0] = -1  # 0
        self.haar_weights[2, 0, 1, 1] = -1
        self.haar_weights[3, 0, 1, 0] = -1  # 45
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x):
        out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
        return F.conv_transpose2d(out, self.haar_weights.to(x.device), bias=None,
                                  stride=2, groups=self.channel_in)


class BDeformConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, bias=None,
                 max_sscale=3, min_sscale=0.5, isReScale=True, init_angle=0):
        super(BDeformConv, self).__init__()
        assert (min_sscale is None and max_sscale is None) or\
               (min_sscale is not None and max_sscale is not None),\
            "Both max_sscale and min_sscale must be set simultaneously or both must be None"
        assert min_sscale < max_sscale, "max_sscale must be larger than min_sscale"

        if isinstance(stride, tuple):
            self.stride = stride
        else:
            self.stride = (stride, stride)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.padding = padding

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding,
                              stride=stride, bias=bias)

        base_points = self.generate_base_points(kernel_size)
        base_points = torch.tensor(base_points).permute(1, 0)
        self.register_buffer('base_offset', base_points[None, None, None, :, :])


        self.conv_rotation = nn.Conv2d(inc, 2, 3, 1, 1)

        self.conv_stretch = nn.Conv2d(inc, 1, 3, 1, 1)

        self.acti = nn.Tanh()

        if max_sscale is not None and min_sscale is not None:
            self.max_sscale = max_sscale
            self.min_sscale = min_sscale
            self.a = (max_sscale - min_sscale) / 2
            self.b = (max_sscale + min_sscale) / 2

        self.isReScale = isReScale
        if self.isReScale:
            self.conv_whole = nn.Conv2d(inc, 1, 3, 1, 1)

        self.init_angle = init_angle
        self.init_offset()

    def init_offset(self):

        nn.init.constant_(self.conv_stretch.weight, 0)
        if self.max_sscale is not None and self.min_sscale is not None:
            tanh_c1 = (2 - self.max_sscale - self.min_sscale) / (self.max_sscale - self.min_sscale)
            tanh_c1 = torch.clamp(torch.tensor(tanh_c1), -0.9999, 0.9999)
            nn.init.constant_(self.conv_stretch.bias, float(torch.atanh(tanh_c1)))
        else:
            nn.init.constant_(self.conv_stretch.bias, 1)
        if self.isReScale:
            nn.init.constant_(self.conv_whole.weight, 0)
            nn.init.constant_(self.conv_whole.bias, 0)
        nn.init.constant_(self.conv_rotation.weight, 0)
        nn.init.constant_(self.conv_rotation.bias[0], math.sin(math.radians(self.init_angle)))
        nn.init.constant_(self.conv_rotation.bias[1], math.cos(math.radians(self.init_angle)))

    def generate_base_points(self, kernel_size):
        kh, kw = kernel_size
        base_points = []
        for i in range(kh):
            for j in range(kw):
                base_points.append([i - kh // 2, j - kw // 2])
        return base_points

    def forward(self, x):
        b, c, h, w = x.shape
        rotation = self.conv_rotation(x)
        sin = rotation[:, 0, :, :]
        cos = rotation[:, 1, :, :]
        norm = torch.sqrt(sin ** 2 + cos ** 2 + 1e-6)
        sin = sin / norm
        cos = cos / norm
        rotation_matrix = torch.stack([
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1)
        ], dim=-2)

        base_offset = self.base_offset.to(x.dtype).expand(b, h, w, -1, -1)
        banded_offset = base_offset.clone()
        if self.max_sscale is not None and self.min_sscale is not None:
            r = self.acti(self.conv_stretch(x)) * self.a + self.b
        else:
            r = self.conv_stretch(x)
        r = r.permute(0, 2, 3, 1)
        banded_offset[..., 0, :] *= r

        if self.isReScale:
            wr = 1 + F.relu(self.conv_whole(x))
            wr = wr.permute(0, 2, 3, 1).unsqueeze(-1)
            banded_offset *= wr

        banded_offset = torch.matmul(rotation_matrix, banded_offset) - base_offset
        banded_offset = banded_offset.permute(0, 4, 3, 1, 2).contiguous().view(b, -1, h, w)

        return deform_conv2d(x, offset=banded_offset, weight=self.conv.weight,
                             bias=self.conv.bias, stride=self.stride,
                             padding=self.padding)


class HaarWaveletProcessing(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(HaarWaveletProcessing, self).__init__()
        self.haar_down = HaarDownsampling(channel_in)
        self.inv_haar = InvHaarDownsampling(channel_out)

        self.low_freq_conv = nn.Conv2d(channel_in * 2, channel_out,
                                       kernel_size=3, stride=1, padding=1)

        self.high_freq_conv_0 = BDeformConv(channel_in * 2, channel_out, kernel_size=3,
                                            stride=1, init_angle=0)
        self.high_freq_conv_45 = BDeformConv(channel_in * 2, channel_out, kernel_size=3,
                                             stride=1, init_angle=45)
        self.high_freq_conv_90 = BDeformConv(channel_in * 2, channel_out, kernel_size=3,
                                             stride=1, init_angle=90)

        self.fuse_high_90 = FuseModule(in_dim=channel_in)
        self.fuse_high_0 = FuseModule(in_dim=channel_in)
        self.fuse_high_45 = FuseModule(in_dim=channel_in)

    def forward(self, x, y):
        haar_out1 = self.haar_down(x)
        haar_out2 = self.haar_down(y)

        low_freq1 = haar_out1[:, :x.shape[1], :, :]
        high_freq_11 = haar_out1[:, x.shape[1]:2 * x.shape[1], :, :]   # 90°
        high_freq_21 = haar_out1[:, 2 * x.shape[1]:3 * x.shape[1], :, :]  # 0°
        high_freq_31 = haar_out1[:, 3 * x.shape[1]:, :, :]              # 45°

        low_freq2 = haar_out2[:, :y.shape[1], :, :]
        high_freq_12 = haar_out2[:, y.shape[1]:2 * y.shape[1], :, :]
        high_freq_22 = haar_out2[:, 2 * y.shape[1]:3 * y.shape[1], :, :]
        high_freq_32 = haar_out2[:, 3 * y.shape[1]:, :, :]

        low_freq = torch.cat([low_freq1, low_freq2], dim=1)
        low_freq_processed = self.low_freq_conv(low_freq)

        fuse_h1_90, fuse_h2_90 = self.fuse_high_90(high_freq_11, high_freq_12)
        high_freq_1 = torch.cat([fuse_h1_90, fuse_h2_90], dim=1)

        fuse_h1_0, fuse_h2_0 = self.fuse_high_0(high_freq_21, high_freq_22)
        high_freq_2 = torch.cat([fuse_h1_0, fuse_h2_0], dim=1)

        fuse_h1_45, fuse_h2_45 = self.fuse_high_45(high_freq_31, high_freq_32)
        high_freq_3 = torch.cat([fuse_h1_45, fuse_h2_45], dim=1)

        high_freq_1_processed = self.high_freq_conv_90(high_freq_1)
        high_freq_2_processed = self.high_freq_conv_0(high_freq_2)
        high_freq_3_processed = self.high_freq_conv_45(high_freq_3)

        processed = torch.cat([low_freq_processed, high_freq_1_processed,
                               high_freq_2_processed, high_freq_3_processed], dim=1)

        return self.inv_haar(processed)
