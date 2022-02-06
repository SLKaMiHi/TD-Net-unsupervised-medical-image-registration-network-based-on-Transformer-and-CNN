import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange

#ResT
class SepConv3d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,):
        super(SepConv3d, self).__init__()
        self.depthwise = torch.nn.Conv3d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         dilation=dilation,
                                         groups=in_channels)
        self.bn = torch.nn.BatchNorm3d(in_channels)
        self.pointwise = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 image_size,
                 num_heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False,
                 kernel_size=3,
                 q_stride=1):
        super().__init__()
        self.num_heads = num_heads
        self.img_size = image_size
        head_dim = dim // num_heads
        pad = (kernel_size - q_stride) // 2
        inner_dim = dim
        self.scale = qk_scale or head_dim ** -0.5

        self.q = SepConv3d(dim, inner_dim, kernel_size, q_stride, pad)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv3d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm3d(self.num_heads)

    def forward(self, x, H, W, L):
        B, N, C = x.shape
        b, n, _, h = *x.shape, self.num_heads
        xq = rearrange(x, 'b (l w d) n -> b n l w d', l=self.img_size[0], w=self.img_size[1], d=self.img_size[2])
        q = self.q(xq)
        q = rearrange(q, 'b (h d) l w k -> b h (l w k) d', h=h)


        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W, L)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, image_size, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, image_size, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W, L):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W, L))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x





class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))




class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, bn, mode='maintain'):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))
        self.bn = bn
        # lightweight method choose.
        self.method = 'multi_conv'

        if mode == 'half':
            stride = 2
        elif mode == 'maintain':
            stride = 1
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1)
        if bn:
            self.bn=nn.BatchNorm3d(out_channels)

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, out):
        """
        Pass the input through the conv_block
            """
        out = self.main(out)
        if self.bn:
            out = self.bn(out)
            print("bn is on")
        out = self.activation(out)
        return out

class Net(nn.Module):
    def __init__(self, vol_size, img_dim, image_size=[48, 56, 48], dim=64, kernels=[3, 3, 3], strides=[2, 2, 2], embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 depths=[2, 2, 2], sr_ratios=[8, 4, 2], bn=False,
                 norm_layer=nn.LayerNorm, apply_transform=False):

        super(Net, self).__init__()
        self.entry_net = conv_block(img_dim, 2, 16, bn)
        self.down1 = conv_block(img_dim, 16, 32, bn, 'half')
        self.dim = dim
        self.pos = PA(32)
        self.depths = depths
        self.size1 = [image_size[0] // 2, image_size[1] // 2, image_size[2] // 2]
        self.size2 = [image_size[0] // 4, image_size[1] // 4, image_size[2] // 4]
        self.size3 = [image_size[0] // 8, image_size[1] // 8, image_size[2] // 8]

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        ##### Stage 1 #######
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv3d(32, dim, kernels[0], strides[0], 1)
            )

        self.stage1_transformer = nn.ModuleList(
            [Block(embed_dims[0], num_heads[0], self.size1, mlp_ratios[0], qkv_bias, qk_scale, drop_rate,
                   attn_drop_rate,
                   drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[0],
                   apply_transform=apply_transform)
             for i in range(self.depths[0])])

        ##### Stage 2 #######
        in_channels = dim
        scale = num_heads[1] // num_heads[0]
        dim = scale * dim
        cur += depths[0]
        self.stage2_conv_embed = nn.Sequential(
            nn.Conv3d(in_channels, dim, kernels[1], strides[1], 1)
            )

        self.stage2_transformer = nn.ModuleList(
            [Block(embed_dims[1], num_heads[1], self.size2, mlp_ratios[1], qkv_bias, qk_scale, drop_rate,
                   attn_drop_rate,
                   drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[1],
                   apply_transform=apply_transform)
             for i in range(self.depths[1])])

        ##### Stage 3 #######
        in_channels = dim
        scale = num_heads[2] // num_heads[1]
        dim = scale * dim
        cur += depths[1]
        self.stage3_conv_embed = nn.Sequential(
            nn.Conv3d(in_channels, dim, kernels[2], strides[2], 1)
            )

        self.stage3_transformer = nn.ModuleList(
            [Block(embed_dims[2], num_heads[2], self.size3, mlp_ratios[2], qkv_bias, qk_scale, drop_rate,
                   attn_drop_rate,
                   drop_path=dpr[cur + i], norm_layer=norm_layer, sr_ratio=sr_ratios[2],
                   apply_transform=apply_transform)
             for i in range(self.depths[2])])


        #decoder
        self.conv1 = conv_block(img_dim, 256, 128, bn, 'maintain')
        self.upsample_layers11 = conv_block(img_dim, 256, 128, bn, 'maintain')
        self.upsample_layers12 = conv_block(img_dim, 128, 64, bn, 'maintain')
        self.upsample_layers21 = conv_block(img_dim, 128, 64, bn, 'maintain')
        self.upsample_layers22 = conv_block(img_dim, 64, 32, bn, 'maintain')
        self.upsample_layers31 = conv_block(img_dim, 64, 32, bn, 'maintain')
        self.upsample_layers32 = conv_block(img_dim, 64, 32, bn, 'maintain')
        self.upsample_layers33 = conv_block(img_dim, 32, 16, bn, 'maintain')
        self.upsample_layers4 = conv_block(img_dim, 32, 16, bn, 'maintain')


        #flow
        conv_fn = getattr(nn, f'Conv{img_dim}d')
        self.flow = conv_fn(16, img_dim, kernel_size=3, padding=1)

        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.spatial_transform = SpatialTransformer(vol_size[2:])




        self.upsample=nn.Upsample(scale_factor=2)

    def forward(self, src, tgt):
        x = torch.cat([src, tgt], 1)
        x_e1 = self.entry_net(x)
        x_e2 = self.down1(x_e1)


        x = self.stage1_conv_embed(x_e2)
        B, _, H, W, L = x.shape
        x = rearrange(x, 'b c h w d  -> b (h w d ) c', h=self.size1[0], w=self.size1[1], d=self.size1[2])
        for blk in self.stage1_transformer:
            x = blk(x, H, W, L)
        x_e3 = rearrange(x, 'b (h w d) c -> b c h w d', h=self.size1[0], w=self.size1[1], d=self.size1[2])

        x = self.stage2_conv_embed(x_e3)
        B, _, H, W, L = x.shape
        x = rearrange(x, 'b c h w d  -> b (h w d ) c', h=self.size2[0], w=self.size2[1], d=self.size2[2])
        for blk in self.stage2_transformer:
            x = blk(x, H, W, L)
        x_e4 = rearrange(x, 'b (h w d) c -> b c h w d', h=self.size2[0], w=self.size2[1], d=self.size2[2])

        x = self.stage3_conv_embed(x_e4)
        B, _, H, W, L = x.shape
        x = rearrange(x, 'b c h w d  -> b (h w d ) c', h=self.size3[0], w=self.size3[1], d=self.size3[2])
        for blk in self.stage3_transformer:
            x = blk(x, H, W, L)
        x_e5 = rearrange(x, 'b (h w d) c -> b c h w d', h=self.size3[0], w=self.size3[1], d=self.size3[2])

        x_e5 = self.conv1(x_e5)

        #decoder
        x_d4 = self.upsample(x_e5)
        x_d4 = torch.cat([x_d4, x_e4], 1)
        x_d4 = self.upsample_layers11(x_d4)
        x_d4 = self.upsample_layers12(x_d4)

        x_d3 = self.upsample(x_d4)
        x_d3 = torch.cat([x_d3, x_e3], 1)
        x_d3 = self.upsample_layers21(x_d3)
        x_d3 = self.upsample_layers22(x_d3)

        x_d2 = self.upsample(x_d3)
        x_d2 = torch.cat([x_d2, x_e2], 1)
        x_d2 = self.upsample_layers31(x_d2)
        x_d2 = torch.cat([x_d2, x_e2], 1)
        x_d2 = self.upsample_layers32(x_d2)
        x_d2 = self.upsample_layers33(x_d2)

        x_d1 = self.upsample(x_d2)
        x_d1 = torch.cat([x_d1, x_e1], 1)
        x_flow = self.upsample_layers4(x_d1)

        flow = self.flow(x_flow)



        y = self.spatial_transform(src, flow)

        return y, flow



class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """

        new_locs = self.grid + flow
        shape = flow.shape[2:]


        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * ((new_locs[:, i, ...] / (shape[i] - 1) - 0.5))

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]  # 最里边这一维的第一列和第0列的数据
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2,1,0]]
            new_locs_construct=new_locs
        return nnf.grid_sample(src, new_locs_construct, align_corners=True, mode=self.mode)
