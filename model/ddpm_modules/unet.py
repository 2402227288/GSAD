import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter
def exists(x):
    return x is not None

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2 # 正弦余弦各占一半
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count # [0,1)
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0)) # [batch,count]
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)# [batch,dim] 位置编码
        return encoding

class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else: # false
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# model
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__() 
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1) # 步幅为2，尺寸减半
 
    def forward(self, x):
        return self.conv(x)


# building block modules
class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim), # 组归一化
            Swish(), # 就相当于激活函数
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        self.noise_func = FeatureWiseAffine(
            time_emb_dim, dim_out, use_affine_level=False)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum( # 使用 torch.einsum 计算 Q 和 K 的点积,.contiguous()保证张量还是连续的
            "bnchw, bncyx -> bnhwyx", query, key # 这里相当于计算量是空间维度的平方,通道维度进行求和 c
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x


    # x_recon = self.denoise_fn(
    #     torch.cat([x_in['LQ'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod) ## 对去噪网络输入x_t与低光图像LQ和alfa连乘积
class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6, # 输入通道是6，3+3
        out_channel=3, # 输出通道
        inner_channel=64, # 64
        norm_groups=32, # GroupNorm的组数，用于归一化操作
        channel_mults=(1, 1, 2, 2, 4),  # 通道倍增系数，用于控制每层通道数
        attn_res=(16), # 16 # 在哪些分辨率下启用注意力机制
        res_blocks=2, # 2  # 每个阶段的残差块数量
        dropout=0,
        with_noise_level_emb=True, #是否使用噪声嵌入
        image_size=128
    ):
        super().__init__()
        if with_noise_level_emb:
            noise_level_channel = inner_channel # 噪声嵌入通道数等于初始特征通道数 64
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),# 使用正弦位置编码
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(), # x * torch.sigmoid(x)
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None


        num_mults = len(channel_mults)
        pre_channel = inner_channel # 初始通道数
        feat_channels = [pre_channel] # 用于记录每个阶段的通道数，便于跳跃连接
        now_res = image_size # 当前分辨率，初始化为输入图像分辨率
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)] # 通道数增加 6->64
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1) ## 是否是最后一个
            use_attn = (now_res in attn_res) # 判断当前分辨率是否启用注意力
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks): # 残差块
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult # 更新初始通道数
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=noise_level_channel, norm_groups=norm_groups, 
                                dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=noise_level_channel, dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups) # 噪声

        self.var_conv = nn.Sequential(*[      ## Pt
            nn.Conv2d(pre_channel, pre_channel, 3, padding=(3//2), bias=True), 
            nn.ELU(), 
            nn.Conv2d(pre_channel, pre_channel, 3, padding=(3//2), bias=True),
            nn.ELU(), 
            nn.Conv2d(pre_channel, 3, 3, padding=(3//2), bias=True),
            nn.ELU()
        ])
        # self.swish = Swish()
    def default_conv(in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), bias=bias)

    def forward(self, x, noise):

        noise_level = self.noise_level_mlp(noise) if exists(self.noise_level_mlp) else None # [8,1,64] 噪声水平
        feats = []

        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, noise_level)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, noise_level)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), noise_level)
            else:
                x = layer(x)
        return self.final_conv(x), self.var_conv(x)
    
    
