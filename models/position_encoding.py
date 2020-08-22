# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # 这种方式是将每个位置映射到角度上，因此有个scale参数，若初始化时没有指定，则默认为0~2π。
    # 使用这种方式编码，就是对行、列的奇偶位置进行正、余弦编码，然后将两者的结果拼接起来，最终排列到所需的维度排列方式。
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        # 角度范围是0~2π
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        # （b,c,h,w）
        x = tensor_list.tensors
        # （b,h,w）
        mask = tensor_list.mask
        assert mask is not None
        # 在模型训练过程与数据处理中的数据处理部分，我讲到过mask代表的是图像哪些位置是padding而来
        # 的，其值为True的部分就是padding的部分，这里取反后得到not_mask，值为True的部分就是图像
        # 真实有效的部分（非padding）。
        not_mask = ~mask
        # 在第一维（列方向）累加
        # (b,h,w)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # 在第二维（行方向）累加
        # (b,h,w)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # 上面在列和行的方向进行累加是为了在下面部分进行归一化操作。
        if self.normalize:
            eps = 1e-6
            # 列方向上做归一化
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            # 行方向上做归一化
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # pow(1000,2i/d)，2i需要在num_pos_feats范围内。因此i为dim_t//2
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # 下面部分代码对应正、余弦编码公式
        # (b,h,w,num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        # (b,h,w,num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t
        # 在最后一维中，偶数维上使用正弦编码，奇数维上使用余弦编码
        # (b,h,w,num_pos_feats//2,2) -> (b,h,w,2*(num_pos_feats//2))
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # (b,h,w,2*num_pos_feats) -> (b,2*num_pos_feats,h,w)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    # 这里默认需要编码的特征图的行、列不超为50，即位置索引在0~50范围内，对每个位置都嵌入到num_pos_feats（默认256）维。
    def __init__(self, num_pos_feats=256):
        super().__init__()
        # 一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。
        # 个人理解：这是一个矩阵类，里面初始化了一个随机矩阵，矩阵的长是字典的大小（第一个参数），宽是用来表示字典中每个元
        # 素的属性向量，向量的维度根据你想要表示的元素的复杂度而定。类实例化之后可以根据字典中元素的下标来查找元素对应的向量。
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        # 均匀分布
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
    # 下面是前向过程，分别对一行和一列中的每个位置进行编码。
    # 在这种方式的编码下，所有行同一列的横坐标（x_emb）编
    # 码结果是一样的，在dim1中处于pos的前num_pos_feats维；同
    # 理，所有列所有列同一行的纵坐标（y_emb）编码结果也是一样
    # 的，在dim1中处于pos的后num_pos_feats维。
    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        # 一行中的每个位置
        i = torch.arange(w, device=x.device)
        # 一列中的每个位置
        j = torch.arange(h, device=x.device)
        # (w,num_pos_feats)
        x_emb = self.col_embed(i)
        # (h,num_pos_feats)
        y_emb = self.row_embed(j)
        # 最后将行、列编码结果拼接起来并扩充第一维，与batch size对应。
        # (h,w,2*num_pos_feats) -> (2*num_pos_featsh,h,w) -> (1,2*num_pos_featsh,h,w) -> (b,2*num_pos_feats,h,w)
        pos = torch.cat([ # dim=-1 竖着上下罗列
            # (h,w,num_pos_feats)
            x_emb.unsqueeze(0).repeat(h, 1, 1), # 沿着指定的维度，对原来的tensor进行数据复制。
            # (h,w,num_pos_feats)
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


# 这部分的实现和Transformer那篇paper中实现的基本类似，有两种方式来实现
# 位置编码，一种是可学习的绝对位置编码，即图像中每个位置绝对地对应一
# 个不同的编码值，且这种编码方式是可学习的；另一种则是使用正、余弦函数来
# 对奇、偶位置进行编码，不需要额外的参数进行学习。
def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        # 余弦编码方式
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        # 可学习的绝对编码方式
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
