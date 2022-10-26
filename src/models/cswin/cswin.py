# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Define CSwin Transformer model"""

import numpy as np

from mindspore import nn
from mindspore import numpy as msnp
from mindspore import ops
from mindspore.common import dtype as mstype
from mindspore.common import initializer as weight_init
from mindspore.common import Parameter, Tensor

from src.models.cswin.misc import _ntuple, Identity, DropPath1D

to_2tuple = _ntuple(2)


class CustomRearrange(nn.Cell):
    """CustomRearrange"""
    def construct(self, x):
        b, c, h, w = x.shape
        # assert h == self.h, "CustomRearrange h shape error"
        # assert w == self.w, "CustomRearrange w shape error"
        x = ops.Transpose()(x, (0, 2, 3, 1))
        x = ops.Reshape()(x, (b, h * w, c))
        return x


class Mlp(nn.Cell):
    """MLP Cell"""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=True)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePEAttention(nn.Cell):
    """LePEAttention"""

    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8,
                 attn_drop=0., proj_drop=0., qk_scale=None):
        super(LePEAttention, self).__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            raise ValueError("ERROR MODE: {}".format(idx))
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=True, group=dim)
        # self.get_v = nn.Conv2d(
        #     dim, dim, kernel_size=3, stride=2, pad_mode="pad", padding=2, has_bias=True, group=dim)
        self.attn_drop = nn.Dropout(1 - attn_drop)

        # operations
        self.batch_matmul_qk = ops.BatchMatMul(transpose_b=True)
        self.batch_matmul_v = ops.BatchMatMul()
        self.reshape = ops.Reshape()
        self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        # self.print = ops.Print()

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = self.resolution
        # H = W = int(N ** 0.5)
        # self.print('im2cswin H&&W: {}'.format(H))
        # print('im2cswin H&&W: {}, resolution: {}'.format(H, self.resolution), flush=True)
        x = self.reshape(self.transpose(x, (0, 2, 1)), (B, C, H, W))

        # img2windows part
        x = self.reshape(x, (B, C, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp))
        x = self.transpose(x, (0, 2, 4, 3, 5, 1))
        # x = self.reshape(x, (-1, H_sp * W_sp, C))
        # img2windows part

        x = self.reshape(x, (-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads))
        x = self.transpose(x, (0, 2, 1, 3))

        return x

    def get_lepe(self, x):
        B, N, C = x.shape
        H = W = self.resolution
        # H = W = int(N ** 0.5)
        # self.print('get_lepe H&&W: {}'.format(H))
        # print('get_lepe H&&W: {}, resolution: {}'.format(H, self.resolution), flush=True)
        x = self.reshape(self.transpose(x, (0, 2, 1)), (B, C, H, W))
        x = self.reshape(x, (B, C, H // self.H_sp, self.H_sp, W // self.W_sp, self.W_sp))
        x = self.transpose(x, (0, 2, 4, 1, 3, 5))
        # B', C, H', W'
        x = self.reshape(x, (-1, C, self.H_sp, self.W_sp))
        # B', C, H', W'
        lepe = self.get_v(x)
        lepe = self.reshape(lepe, (-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp))
        lepe = self.transpose(lepe, (0, 1, 3, 2))

        x = self.reshape(x, (-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp))
        x = self.transpose(x, (0, 1, 3, 2))

        return x, lepe

    def construct(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]
        # img2window
        H = W = self.resolution
        B, L, C = q.shape
        # assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v)

        q = q * self.scale
        # batch_matmul already set transpose_b
        attn = self.batch_matmul_qk(q, k)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = self.batch_matmul_v(attn, v) + lepe
        # B head N N @ B head N C
        x = self.reshape(self.transpose(x, (0, 2, 1, 3)), (-1, self.H_sp * self.W_sp, C))

        # window2img
        S = x.shape[0]
        # B_w = int(S / (H * W / self.H_sp / self.W_sp))
        B_w = S // (H * W / self.H_sp / self.W_sp)
        x = self.reshape(x, (B_w, H // self.H_sp, W // self.W_sp, self.H_sp, self.W_sp, -1))
        x = self.transpose(x, (0, 1, 3, 2, 4, 5))
        x = self.reshape(x, (B_w, H, W, -1))

        x = self.reshape(x, (B, -1, C))

        return x


class CSWinBlock(nn.Cell):
    """CSWinBlock"""

    def __init__(self, dim, reso, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super(CSWinBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio

        self.qkv = nn.Dense(in_channels=dim, out_channels=dim * 3, has_bias=qkv_bias, activation=None)

        self.norm1 = norm_layer((dim,) if isinstance(dim, int) else dim, epsilon=1e-5)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2

        self.proj = nn.Dense(in_channels=dim, out_channels=dim, has_bias=True, activation=None)
        self.proj_drop = nn.Dropout(1 - drop)

        if last_stage:
            self.attns = nn.CellList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for _ in range(self.branch_num)
            ])
        else:
            self.attns = nn.CellList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)
            ])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer((dim,) if isinstance(dim, int) else dim, epsilon=1e-5)

        # operations
        self.concat = ops.Concat(axis=2)
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x):
        """x: B, H*W, C"""
        H = W = self.patches_resolution
        B, L, C = x.shape
        # assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.transpose(self.reshape(self.qkv(img), (B, -1, 3, C)), (2, 0, 1, 3))
        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C//2])
            x2 = self.attns[1](qkv[:, :, :, C//2:])
            attened_x = self.concat((x1, x2))
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class MergeBlock(nn.Cell):
    """MergeBlock"""
    
    def __init__(self, dim, dim_out, resolution, norm_layer=nn.LayerNorm):
        super(MergeBlock, self).__init__()
        self.resolution = resolution
        self.conv = nn.Conv2d(
            in_channels=dim, out_channels=dim_out, kernel_size=3, stride=2,
            pad_mode="pad", padding=1, has_bias=True)
        if isinstance(dim_out, int):
            dim_out = (dim_out,)
        self.norm = norm_layer(dim_out, epsilon=1e-5)

        # operations
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        # self.print = ops.Print()

    def construct(self, x):
        B, new_HW, C = x.shape
        H = W = self.resolution
        # H = W = int(new_HW ** 0.5)
        # self.print('MergeBlock H&&W: {}'.format(H))
        # print('MergeBlock H&&W: {}, resolution: {}'.format(H, self.resolution), flush=True)
        x = self.reshape(self.transpose(x, (0, 2, 1)), (B, C, H, W))
        x = self.conv(x)
        B, C = x.shape[:2]
        x = self.transpose(self.reshape(x, (B, C, -1)), (0, 2, 1))
        x = self.norm(x)

        return x


class CSWinTransformer(nn.Cell):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96,
                depth=[2,2,6,2], split_size=[3,5,7], num_heads=[2,4,8,16], mlp_ratio=4.,
                qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super(CSWinTransformer, self).__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        heads = num_heads

        self.stage1_conv_embed = nn.SequentialCell(
            nn.Conv2d(
                in_channels=in_chans, out_channels=embed_dim, kernel_size=7, stride=4,
                pad_mode="pad", padding=2, has_bias=True),
            CustomRearrange(),
            norm_layer((embed_dim, ) if isinstance(embed_dim, int) else embed_dim, epsilon=1e-5)
        )

        dpr = np.linspace(0, drop_path_rate, sum(depth)).tolist()

        curr_dim = embed_dim
        self.stage1 = nn.CellList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size//4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.merge1 = MergeBlock(curr_dim, curr_dim*2, resolution=img_size//4)

        curr_dim = curr_dim * 2
        self.stage2 = nn.CellList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size//8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depth[:1])+i], norm_layer=norm_layer)
            for i in range(depth[1])])
        self.merge2 = MergeBlock(curr_dim, curr_dim * 2, resolution=img_size//8)

        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size//16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depth[:2])+i], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.stage3 = nn.CellList(temp_stage3)
        self.merge3 = MergeBlock(curr_dim, curr_dim * 2, resolution=img_size//16)

        curr_dim = curr_dim * 2
        self.stage4 = nn.CellList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
            for i in range(depth[-1])])

        self.norm = norm_layer((curr_dim,) if isinstance(curr_dim, int) else curr_dim, epsilon=1e-5)
        # Classifier head
        if num_classes > 0:
            self.head = nn.Dense(
                in_channels=curr_dim, out_channels=num_classes, has_bias=True, activation=None)
        else:
            self.head = Identity()

        # operations
        self.mean = ops.ReduceMean(keep_dims=False)

        self.init_weight()

    def init_weight(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02), cell.weight.shape,
                                            cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(
                        weight_init.initializer(weight_init.Zero(), cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(
                    weight_init.initializer(weight_init.One(), cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(
                    weight_init.initializer(weight_init.Zero(), cell.beta.shape, cell.beta.dtype))

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def construct_features(self, x):
        B = x.shape[0]
        x = self.stage1_conv_embed(x)
        for blk in self.stage1:
            x = blk(x)

        x = self.merge1(x)
        for blk in self.stage2:
            x = blk(x)

        x = self.merge2(x)
        for blk in self.stage3:
            x = blk(x)

        x = self.merge3(x)
        for blk in self.stage4:
            x = blk(x)

        x = self.norm(x)

        return self.mean(x, 1)

    def construct(self, x):
        x = self.construct_features(x)
        x = self.head(x)
        return x


def CSWin_64_24322_small_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        patch_size=4, embed_dim=64, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4., **kwargs)

    return model


def CSWin_144_24322_large_224(pretrained=False, **kwargs):
    model = CSWinTransformer(
        patch_size=4, embed_dim=144, depth=[2,4,32,2],
        split_size=[1,2,7,7], num_heads=[6,12,24,24], mlp_ratio=4., **kwargs)

    return model


def CSWin_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2],
        split_size=[1,2,12,12], num_heads=[4,8,16,32], mlp_ratio=4., **kwargs)
    return model


def main():
    from mindspore import context
    # context.set_context(mode=context.PYNATIVE_MODE)
    context.set_context(mode=context.GRAPH_MODE)

    x = Tensor(np.random.rand(2, 3, 224, 224), dtype=mstype.float32)
    model = CSWin_64_24322_small_224(drop_path_rate=0.4)
    # model = CSWin_144_24322_large_224(drop_path_rate=0.2)

    # x = Tensor(np.random.rand(2, 3, 384, 384), dtype=mstype.float32)
    # model = CSWin_96_24322_base_384(img_size=384, drop_path_rate=0.5)
    # print("====== model ======\n{}".format(model), flush=True)

    y = model(x)
    print(y.shape, flush=True)


if __name__ == "__main__":
    main()
