import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from BDeformConv_layer4 import HaarDownsampling, InvHaarDownsampling, FuseModule


_sam_perturb_state = {"alpha": 0.0, "seed": 0, "step": 0}


def set_sam_perturb(alpha, seed=0):

    _sam_perturb_state["alpha"] = float(alpha)
    _sam_perturb_state["seed"] = int(seed)
    _sam_perturb_state["step"] = 0


def get_sam_perturb_alpha():
    return _sam_perturb_state["alpha"]


def _perturb_sam_entry(sam_features):

    alpha = _sam_perturb_state["alpha"]
    if alpha <= 0:
        return sam_features
    g = torch.Generator(device=sam_features.device)
    g.manual_seed(_sam_perturb_state["seed"] + _sam_perturb_state["step"])
    _sam_perturb_state["step"] += 1
    noise = torch.randn(sam_features.shape, generator=g,
                        device=sam_features.device, dtype=sam_features.dtype)
    return sam_features + alpha * sam_features.std() * noise


class SDADC(nn.Module):
    def __init__(self, inc, sam_channels, outc, kernel_size=3, stride=1, bias=None,
                 max_sscale=3, min_sscale=0.5, isReScale=True, init_angle=0):
        super(SDADC, self).__init__()

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


        self.conv_rotation = nn.Conv2d(inc, 2, 3, 1, 1)
        self.conv_stretch = nn.Conv2d(inc, 1, 3, 1, 1)
        if isReScale:
            self.conv_whole = nn.Conv2d(inc, 1, 3, 1, 1)


        self.sam_preprocess = nn.Sequential(
            nn.Conv2d(sam_channels, inc, 1, bias=False),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True)
        )


        self.sam_spatial_attention = nn.Sequential(
            nn.Conv2d(inc + inc, inc // 4, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(inc // 4, 1, 1),
            nn.Sigmoid()
        )


        self.sam_feature_enhance = nn.Sequential(
            nn.Conv2d(inc * 2, inc, 3, 1, 1),
            nn.BatchNorm2d(inc),
            nn.ReLU(inplace=True)
        )


        self.sam_offset_guide = nn.Sequential(
            nn.Conv2d(inc, kernel_size[0] * kernel_size[1] * 2, 3, 1, 1),
            nn.Tanh()
        )


        self.offset_fusion_weight = nn.Parameter(torch.tensor(0.1))

        self.acti = nn.Tanh()
        if max_sscale is not None and min_sscale is not None:
            self.max_sscale = max_sscale
            self.min_sscale = min_sscale
            self.a = (max_sscale - min_sscale) / 2
            self.b = (max_sscale + min_sscale) / 2

        self.isReScale = isReScale
        self.init_angle = init_angle

        base_points = self.generate_base_points(kernel_size)
        base_points = torch.tensor(base_points).permute(1, 0)
        self.register_buffer('base_offset', base_points[None, None, None, :, :])

        self.init_offset()

    def init_offset(self):

        nn.init.constant_(self.conv_stretch.weight, 0)
        nn.init.constant_(self.conv_rotation.weight, 0)
        nn.init.constant_(self.conv_rotation.bias[0], math.sin(math.radians(self.init_angle)))
        nn.init.constant_(self.conv_rotation.bias[1], math.cos(math.radians(self.init_angle)))

        if hasattr(self, 'max_sscale') and hasattr(self, 'min_sscale'):
            tanh_c1 = (2 - self.max_sscale - self.min_sscale) / (self.max_sscale - self.min_sscale)
            tanh_c1 = torch.clamp(torch.tensor(tanh_c1), -0.9999, 0.9999)
            nn.init.constant_(self.conv_stretch.bias, float(torch.atanh(tanh_c1)))
        else:
            nn.init.constant_(self.conv_stretch.bias, 1)

        if self.isReScale:
            nn.init.constant_(self.conv_whole.weight, 0)
            nn.init.constant_(self.conv_whole.bias, 0)

    def generate_base_points(self, kernel_size):
        kh, kw = kernel_size
        base_points = []
        for i in range(kh):
            for j in range(kw):
                base_points.append([i - kh // 2, j - kw // 2])
        return base_points

    def forward(self, x, sam_features=None, return_debug=False):

        b, c, h, w = x.shape

        if sam_features is not None:
            if sam_features.shape[-2:] != x.shape[-2:]:
                sam_features = F.interpolate(sam_features, size=(h, w),
                                             mode='bilinear', align_corners=False)

            sam_aligned = self.sam_preprocess(sam_features)

            combined_feat = torch.cat([x, sam_aligned], dim=1)
            spatial_attention = self.sam_spatial_attention(combined_feat)

            attended_sam = sam_aligned * spatial_attention
            enhanced_x = self.sam_feature_enhance(torch.cat([x, attended_sam], dim=1))


            input_for_offset = enhanced_x

            sam_guided_offset = self.sam_offset_guide(attended_sam)
            sam_guided_offset = sam_guided_offset.view(b, -1, 2, h, w).permute(0, 3, 4, 2, 1)
        else:
            input_for_offset = x
            sam_guided_offset = None

        rotation = self.conv_rotation(input_for_offset)
        sin = rotation[:, 0, :, :]
        cos = rotation[:, 1, :, :]
        norm = torch.sqrt(sin ** 2 + cos ** 2 + 1e-6)
        sin = sin / norm
        cos = cos / norm

        rotation_matrix = torch.stack([
            torch.stack([cos, sin], dim=-1),
            torch.stack([-sin, cos], dim=-1)
        ], dim=-2)

        if hasattr(self, 'max_sscale') and hasattr(self, 'min_sscale'):
            r = self.acti(self.conv_stretch(input_for_offset)) * self.a + self.b
        else:
            r = self.conv_stretch(input_for_offset)
        r = r.permute(0, 2, 3, 1)

        base_offset = self.base_offset.to(x.dtype).expand(b, h, w, -1, -1)
        banded_offset = base_offset.clone()
        banded_offset[..., 0, :] *= r

        if self.isReScale:
            wr = 1 + F.relu(self.conv_whole(input_for_offset))
            wr = wr.permute(0, 2, 3, 1).unsqueeze(-1)
            banded_offset *= wr

        banded_offset = torch.matmul(rotation_matrix, banded_offset)
        final_offset = banded_offset - base_offset

        if sam_guided_offset is not None:
            final_offset += self.offset_fusion_weight * sam_guided_offset

        final_offset = final_offset.permute(0, 4, 3, 1, 2).contiguous().view(b, -1, h, w)

        out = deform_conv2d(x, offset=final_offset, weight=self.conv.weight,
                            bias=self.conv.bias, stride=self.stride, padding=self.padding)
        if return_debug:
            debug = {
                "final_offset": final_offset.detach(),          # [B, 2K, H, W]
                "sam_guided_offset": (sam_guided_offset.detach()
                                      if sam_guided_offset is not None else None),
                "spatial_attention": (spatial_attention.detach()
                                      if sam_features is not None else None),
            }
            return out, debug
        return out


class FTSCM(nn.Module):
    def __init__(self, channel_in, channel_out, sam_channels=256, use_cdim=True,
                 init_angle=None, init_angles=None):
        super(FTSCM, self).__init__()
        self.use_cdim = use_cdim
        if init_angles is not None:
            assert len(init_angles) == 3 and all(
                isinstance(a, (int, float)) for a in init_angles)
            a90, a0, a45 = [float(a) for a in init_angles]
        else:
            a0 = 0 if init_angle is None else init_angle
            a45 = 45 if init_angle is None else init_angle
            a90 = 90 if init_angle is None else init_angle

        self.haar_down = HaarDownsampling(channel_in)
        self.inv_haar = InvHaarDownsampling(channel_out)


        self.low_freq_conv = nn.Conv2d(channel_in * 2, channel_out,
                                       kernel_size=3, stride=1, padding=1)


        self.high_freq_conv_0 = SDADC(
            inc=channel_in * 2, sam_channels=sam_channels, outc=channel_out,
            kernel_size=3, stride=1, init_angle=a0
        )
        self.high_freq_conv_45 = SDADC(
            inc=channel_in * 2, sam_channels=sam_channels, outc=channel_out,
            kernel_size=3, stride=1, init_angle=a45
        )
        self.high_freq_conv_90 = SDADC(
            inc=channel_in * 2, sam_channels=sam_channels, outc=channel_out,
            kernel_size=3, stride=1, init_angle=a90
        )


        self.fuse_high_90 = FuseModule(in_dim=channel_in) if use_cdim else None
        self.fuse_high_0 = FuseModule(in_dim=channel_in) if use_cdim else None
        self.fuse_high_45 = FuseModule(in_dim=channel_in) if use_cdim else None


        self.sam_downsample = nn.Sequential(
            nn.Conv2d(sam_channels, sam_channels, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(sam_channels, sam_channels, 3, 1, 1)
        )

    def forward(self, x, y, sam_features=None, return_debug=False):

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

        sam_feat_downsampled = None
        if sam_features is not None:
            sam_feat_downsampled = self.sam_downsample(sam_features)

            sam_feat_downsampled = _perturb_sam_entry(sam_feat_downsampled)

        low_freq = torch.cat([low_freq1, low_freq2], dim=1)
        low_freq_processed = self.low_freq_conv(low_freq)

        if self.use_cdim:
            fuse_h1_90, fuse_h2_90 = self.fuse_high_90(high_freq_11, high_freq_12)
            fuse_h1_0, fuse_h2_0 = self.fuse_high_0(high_freq_21, high_freq_22)
            fuse_h1_45, fuse_h2_45 = self.fuse_high_45(high_freq_31, high_freq_32)
        else:
            fuse_h1_90, fuse_h2_90 = high_freq_11, high_freq_12
            fuse_h1_0, fuse_h2_0 = high_freq_21, high_freq_22
            fuse_h1_45, fuse_h2_45 = high_freq_31, high_freq_32
        high_freq_1 = torch.cat([fuse_h1_90, fuse_h2_90], dim=1)
        high_freq_2 = torch.cat([fuse_h1_0, fuse_h2_0], dim=1)
        high_freq_3 = torch.cat([fuse_h1_45, fuse_h2_45], dim=1)

        if return_debug:
            high_freq_1_processed, dbg90 = self.high_freq_conv_90(
                high_freq_1, sam_feat_downsampled, return_debug=True)
            high_freq_2_processed, dbg0 = self.high_freq_conv_0(
                high_freq_2, sam_feat_downsampled, return_debug=True)
            high_freq_3_processed, dbg45 = self.high_freq_conv_45(
                high_freq_3, sam_feat_downsampled, return_debug=True)
        else:
            high_freq_1_processed = self.high_freq_conv_90(high_freq_1, sam_feat_downsampled)
            high_freq_2_processed = self.high_freq_conv_0(high_freq_2, sam_feat_downsampled)
            high_freq_3_processed = self.high_freq_conv_45(high_freq_3, sam_feat_downsampled)

        processed = torch.cat([
            low_freq_processed,
            high_freq_1_processed,
            high_freq_2_processed,
            high_freq_3_processed
        ], dim=1)

        out = self.inv_haar(processed)
        if return_debug:
            debug = {
                "offset_90": dbg90["final_offset"],
                "offset_0": dbg0["final_offset"],
                "offset_45": dbg45["final_offset"],
                "attn_90": dbg90["spatial_attention"],
                "attn_0": dbg0["spatial_attention"],
                "attn_45": dbg45["spatial_attention"],
            }
            return out, debug
        return out
