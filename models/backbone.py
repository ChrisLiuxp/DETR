# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        # 在实现的时候，需要将以下4个量注册到buffer，以便阻止梯度反向传播而更新它们，同时又能够记录在模型的state_dict中。
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        # 若return_interm_layers设置为True，则需要记录每一层（ResNet的layer）的输出。
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        # IntermediateLayerGetter 这个类是在torchvision中实现的，它继承nn.ModuleDict，接收一
        # 个nn.Module和一个dict作为初始化参数，dict的key对应nn.Module的模块，value则是用户自定
        # 义的对应各个模块输出的命名，官方给出的例子如下：
        # Examples::
        #         >>> m = torchvision.models.resnet18(pretrained=True)
        #         >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        #         >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        #         >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        #         >>> out = new_m(torch.rand(1, 3, 224, 224))
        #         >>> print([(k, v.shape) for k, v in out.items()])
        #         >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        #         >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    # 来看看BackboneBase的前向过程。self.body就是上述提到的IntermediateLayerGetter，它
    # 的输出是一个dict，对应了每层的输出，key是用户自定义的赋予输出特征图的名字。
    # BackboneBase的前向方法中的输入是NestedTensor这个类的实例，其实质就是将图像张量和对应的mask封装到一起。
    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            # 将mask插值到与输出特征图尺寸一致
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out

# 继承BackboneBase这个类，实际的backbone是使用torchvision里实现的resnet
class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        # pretrained=is_main_process() 代表仅在主进程中使用预训练权重
        # norm_layer=FrozenBatchNorm2d，代表这里使用的归一化层是FrozenBatchNorm2d，
        # 这个nn.Module与batch normalization的工作原理类似，只不过将统计量（均值与
        # 方差）和可学习的仿射参数固定住，doc string里的描述是：BatchNorm2d where
        # the batch statistics and the affine parameters are fixed.
        # getattr() 函数用于返回一个对象属性值。getattr(object, name[, default])。参数object -- 对象。name -- 字符串，对象属性。default -- 默认返回值，如果不提供该参数，在没有对应属性时，将触发 AttributeError。
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

"""
Joiner就是将backbone和position encoding集成的一个nn.Module里，使得前向过程中可以使用两者的功能。
Joiner是nn.Sequential的子类，通过初始化，使得self[0]是backbone，self[1]是position encoding。
前向过程就是对backbone的每层输出都进行位置编码，最终返回backbone的输出及对应的位置编码结果。
"""
class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # backbone的输出
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos

# 将位置编码与backbone集成到一起作为一个model，在backbone输出特征图的同时对其进行位置编码，以便后续Transformer使用
# backbone的构建通过bulid_backbone这个方法封装，主要做的就是分别构建位置编码部分与backbone，然后将两者封装到一个
# nn.Module里，在前向过程中实现两者的功能。（重点应该是在位置编码的部分，backbone部分毕竟是调用torchvision的内置模型）
def build_backbone(args):
    # 对backbone输出的特征图进行位置编码，用于后续Transformer部分
    position_embedding = build_position_encoding(args)
    # 是否需要训练backbone（即是否采用预训练backbone）
    train_backbone = args.lr_backbone > 0
    # 是否需要记录backbone的每层输出
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    # 将backbone和位置编码集合在一个model
    model = Joiner(backbone, position_embedding)
    #
    model.num_channels = backbone.num_channels
    return model
