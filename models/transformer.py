# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# 发现它不仅MultiheadAttention做了改动，transformer也改动了，因此直接引用还不够，转置lite transformer
# from .multihead_attention import MultiheadAttention
"""
1、通过Decoder和Encoder的实现可以发现，作者极力强调位置嵌入的作用，每次在self-attention操作时都伴随着position embedding，
因为Transformer本身是permute invariant，即对排列和位置是不care的，而我们很清楚，在detection任务中，位置信息有着举足轻重的地位！
2、另外，看完DecoderLayer的实现，会发现可能存在“重复归一化”的问题。当使用后归一化的前向方式时，每个DecoderLayer
的输出是归一化后的结果，但是在Decoder的前向过程中会使用self.norm对其再进行一次归一化！
3、Transformer其输出结果并不是最终的预测结果，还需要进行转换
"""
class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        # 构建Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        # 如果是后归一化的方式，那么Encoder每层输出都会进行归一化，因此再Encoder对最后的输出就不需要再额外进行归一化了，这种情况下就将encoder_norm设置为None
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        # 构建Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        # 注意在Decoder中decoder_norm始终存在，哈哈哈，怕报错
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        # 这里C与hidden_dim相等（Transformer的前向过程如下，首先是对输入做reshape，这里的src是已经将CNN提取的特征维度映射到hidden_dim的结果。）
        bs, c, h, w = src.shape
        # (h*w，bs，c=hidden_dim)
        src = src.flatten(2).permute(2, 0, 1)  # 我理解的这个flatten(2)是把下标为2和以后的维，压成一个维
        # (h*w，bs，c=hidden_dim)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # (num_queries, bs, hidden_dim)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        # (bs, h*w)
        mask = mask.flatten(1)
        # query embedding 有点anchor的味道，而且是自学习的anchor，作者使用了nn.Embedding实现，num_queries代表图像中
        # 有多少个目标（位置），默认是100个，对这些目标（位置）全部进行嵌入，维度映射到hidden_dim.将query_embedding的
        # 权重作为参数输入到Transformer的前向过程，作为position encoding。而与这个position encoding相结合形成嵌入的
        # 是什么东东呢？当然是我们需要预测的目标咯！那么我都还不知道这些目标是什么在哪里，如何将它实体化？于是作者就直接
        # 将它初始化为全0，shape和query embedding 的权重一致。
        # (num_queries, bs, hidden_dim)
        tgt = torch.zeros_like(query_embed)
        # 然后就是将以上输入参数依次送进Encoder和Decoder，最后再reshape到需要的结果。
        # (h*w, bs, c=hidden_dim)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # 在项目整体代码中，TransformerDecoder的初始化参数return_intermediate设置为true(在最下面的build_transformer方法中)，
        # 因此Decoder的输出包含了每层的结果，shape是(6, num_queries, bs, hidden_dim)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        # (6, bs, num_queries, hidden_dim), (bs, c=hidden_dim, h, w)
        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w) # 瞅这意思是transpose(1, 2)把下标是1和2的交换了；view()和reshape()功能类似

"""
Encoder的每层是TransformerEncoderLayer的一个实例
"""
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # Encoder通常有6层，每层结构相同，这里使用_get_clones()这个方法将结构相同的层复制多次，返回一个nn.ModuleList实例。
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # 归一化层
        self.norm = norm
    # Encoder的前向过程如下，循坏调用每层的前向过程即可，前一层的输出作为后一层的输入。最后，
    # 若指定了需要归一化，那么就对最后一层的输出作归一化。
    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # src对应backbone最后一层输出的特征图，并且维度映射到了hidden_dim，shape是（h*w,b,hidden_dim）
        # pos对应backbone最后一层输出的特征图对应的位置编码，shape是（h*w,b,c）
        # src_key_padding_mask对应backbone最后一层输出的特征图对应的mask，shape是（b,h*w）
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # 是否需要记录中间每层的结果（比原来的transformer新增的）
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # 其前向过程如下。具体操作和Encoder的也类似，只不过需要先将以下参数梳理清楚，之后整个代码看起来就十分好理解了。
        # tgt是query embedding，shape是(num_queries,b,hidden_dim)
        # query_pos是对应tgt的位置编码，shape和tgt一致
        # memory是Encoder的输出，shape是(h*w,b,hidden_dim)
        # memory_key_padding_mask对应Encoder的src_key_padding_mask，也是EncoderLayer的key_padding_mask，shape是(b,h*w)
        # pos对应输入到Encoder的位置编码，这里代表memory的位置编码，shape和memory一致
        output = tgt

        intermediate = []

        for layer in self.layers:
            # intermediate中记录的是每层输出后的归一化结果，而每一层的输入是前一层输出（没有归一化）的结果。
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                # intermediate中记录的是每层输出后的归一化结果，而每一层的输入是前一层输出（没有归一化）的结果。
                intermediate.append(self.norm(output))
        #不知道你们发现了不，这里的实现有点“令人不舒服”。self.norm是通过初始化时传进来的参数norm（默认为None）设
        # 置的，那么self.norm就有可能是norm，因此下面第一句代码也对此作了判断。但是在上一句中，却在没有作判断的情况下直
        # 接码出了self.norm(output)这句，所以有可能会引发异常。在整体项目代码中，作者在构建Decoder使始终传了norm参数，
        # 使得其不为None，因此不会引发异常。但就单独看Decoder的这部分实现来说，确实是有问题的，如果朋友们直接拿这部分去用，需要注意下这点。
        if self.norm is not None:
            output = self.norm(output)
            # 感觉以下三行部分是“多此一举”，因为本身intermediate记录的就是每层输出的归一化结果了。
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        # 原版transformer没有这个.unsqueeze(0)
        return output.unsqueeze(0)

"""
其主要由多头自注意力层（Multi-Head Self-Attention）和前向反馈层（FFN）构成，另外还包含
了归一化层、激活层、Dropout层以及残差连接。
"""
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # 前向反馈层FNN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # 这里使用了nn.LayerNorm，即层归一化。与BatchNorm（批归一化）不同的是，后者在通道这个维度上进行归一
        # 化，而前者可以指定对最后的哪几个维度进行归一化。另外，其可学习参数γ和β是对应做归一化的那几个维度上
        # 的每个元素的，而非批归一化那种一个通道共享一对γ和β标量。还有一点与批归一化不同，就是其在测试过程中也会计算统计量
        # 官方给出的说明如下：
        # Unlike Batch Normalization and Instance Normalization, which applies scalar scale and bias for each entire channel/plane
        # with the :attr:`affine` option, Layer Normalization applies per-element scale and with :attr:`elementwise_affine`.
        # This layer uses statistics computed from input data in both training and evaluation modes
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # _get_activation_fn()方法根据输入参数指定的激活方式返回对应的激活层，默认是ReLU。
        self.activation = _get_activation_fn(activation)
        # 是否在输入多头自注意层/前向反馈曾前进行归一化
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # 位置嵌入=序列+位置编码
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 在输入多自注意力层和前向反馈层输出后归一化
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    # 先进行归一化的前向过程
    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # 输入多头自注意力层和前向反馈层前先进行归一化
        src2 = self.norm1(src)
        # 在输入多头自注意力层时需要先进行位置嵌入，即结合位置编码。（这个与原版的不太一样，原版没有这个）
        q = k = self.with_pos_embed(src2, pos)
        # self.self_attn是nn.MultiheadAttention的实例，其前向过程返回两部分，第一个是自注意力层的输出，第二个是自注意力权重，因此可以看到这里取了输出索引为0的部分。
        # key_padding_mask对应上述Encoder的src_key_padding_mask，是backbone最后一层输出特征图对应的mask，值为True的那些部分是原始图像padding的部分，在生成注意力
        # 的过程中会被填充为-inf，最终生成注意力需要经过softmax时会被忽略不计。官方对该参数的解释如下：
        # key_padding_mask – if provided, specified padding elements in the key will be ignored by the attention.
        # This is an binary mask. When the value is True, the corresponding value on the attention layer will be filled with -inf.
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # EncoderLayer的前向过程分为两种情况，一种是在输入多头自注意力层和前向反馈层前先进行归一化，另一种则是在这两个层输出后再进行归一化操作。
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # 多头自注意力层
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Encoder-Decoder Layer（DecoderLayer与Encoder的实现类似，只不过多了一层 Encoder-Decoder Layer，
        # 其实质也是多头自注意力层，但是key和value来自于Encoder的输出。）
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # 前向反馈FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # 分别对应于以上三个层
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # 首先进行位置嵌入（这个与原版的不太一样，原版没有这个）
        q = k = self.with_pos_embed(tgt, query_pos)
        # 多头自注意力层，输入参数不包含Encoder的输出（在第一个多头自注意层中，输入均和Encoder无关。）
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        # 该层后进行归一化
        tgt = self.norm1(tgt)
        # 第二个多头自注意力层，Encoder-Decoder层，key和value来自Encoder的输出，query来自上一层的输出，注意query和key需要进行位置嵌入（这块不理解可以对着模型图看）
        # memory_key_padding_mask对应EncoderLayer的key_padding_mask
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        # 该层输出后在归一化
        tgt = self.norm2(tgt)
        # 前向反馈层FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        # 该层输出后再归一化
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # DecoderLayer的前向过程也如同EncoderLayer般分为两种情况
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
