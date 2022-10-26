# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Get SwinTransformer of different size for args"""
from .cswin import CSWinTransformer
# from .cswin import CSWin_64_24322_small_224


def get_cswintransformer(args):
    """get cswintransformer according to args"""
    # override args
    image_size = args.image_size
    patch_size = args.patch_size
    in_chans = args.in_channel
    num_classes = args.num_classes
    embed_dim = args.embed_dim
    depth = args.depth
    split_size = args.split_size
    num_heads = args.num_heads
    mlp_ratio = args.mlp_ratio
    drop_path_rate = args.drop_path_rate

    qkv_bias = True
    qk_scale = None
    drop_rate = 0.
    attn_drop_rate = 0.

    # print(25 * "=" + "MODEL CONFIG" + 25 * "=")
    # print(f"==> IMAGE_SIZE:         {image_size}")
    # print(f"==> PATCH_SIZE:         {patch_size}")
    # print(f"==> NUM_CLASSES:        {args.num_classes}")
    # print(f"==> EMBED_DIM:          {embed_dim}")
    # print(f"==> NUM_HEADS:          {num_heads}")
    # print(f"==> DEPTHS:             {depths}")
    # print(f"==> WINDOW_SIZE:        {window_size}")
    # print(f"==> MLP_RATIO:          {mlp_ratio}")
    # print(f"==> QKV_BIAS:           {qkv_bias}")
    # print(f"==> QK_SCALE:           {qk_scale}")
    # print(f"==> DROP_PATH_RATE:     {drop_path_rate}")
    # print(f"==> APE:                {ape}")
    # print(f"==> PATCH_NORM:         {patch_norm}")
    # print(25 * "=" + "FINISHED" + 25 * "=")

    model = CSWinTransformer(
        img_size=image_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
        embed_dim=embed_dim, depth=depth, split_size=split_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate)

    # print(model)
    return model


def cswin_64_24322_small_224(args):
    """cswin_64_24322_small_224"""
    return get_cswintransformer(args)


def cswin_144_24322_large_224(args):
    return get_cswintransformer(args)
