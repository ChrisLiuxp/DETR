# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from .transformer import build_transformer


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    # num_classes不包含背景
    # aux_loss代表是否要对Transformer中Decoder的每层输出都计算loss
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        # class_embed生成分类的预测结果，最后一维对应物体类别数量，并且加上背景这一类
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # bbox_embed生成回归的预测结果
        # MLP就是多层感知机的缩写，顾名思义，由多层nn.Linear()组成，这里有3层线性层，中间每层的
        # 维度被映射到hidden_dim，最后一层维度映射为4，代表bbox的中心点横、纵坐标和宽、高。
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # query embedding 有点anchor的味道，而且是自学习的anchor，作者使用了nn.Embedding实现，num_queries代表图像中
        # 有多少个目标（位置），默认是100个，对这些目标（位置）全部进行嵌入，维度映射到hidden_dim
        # query_embed用于在Transformer中对初始化query以及对其编码生成嵌入
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # input_proj是将CNN提取的特征维度映射到Transformer隐层的维度，转化为序列
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

    # DETR的前向过程。输入是一个NestedTensor类的对象（NestedTensor就是加上mask的tensor）
    # 前向过程可分解为两部分，先利用CNN提取特征，然后将特征图映射为序列形式，最后输入Transformer进行编、解码得到输出结果。
    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # 首先将样本转换为NestedTensor类型
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # 输入到CNN提取特征
        features, pos = self.backbone(samples)

        # 取出最后一层特征及对应的mask
        src, mask = features[-1].decompose()
        assert mask is not None
        # 将query_embedding的权重作为参数输入到Transformer的前向过程，作为position encoding。而与这个position encoding相结
        # 合形成嵌入的是什么东东呢？当然是我们需要预测的目标咯！那么我都还不知道这些目标是什么在哪里，如何将它实体化？于是作者就
        # 直接将它初始化为全0，shape和query embedding 的权重一致。（可在Transformer前向传播过程中看到）
        # Transformer的输出是元组，分别为Decoder和Encoder的输出，因此在这里取第一个代表的是Decoder的输出
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        # 第二部分就是对输出的维度进行转化，与分类和回归任务所要求的相对应。
        # 生成分类与回归的预测结果
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()
        # 由于hs包含了Transformer中Decoder每层的输出，因此索引为-1代表去掉最后一层的输出
        # 那么这里问题就来了，在测试的时候，预测结果就有num_queries（默认100）个，而图片中实际的物体数量通常并没有那么多，这时候
        # 该如何处理呢？如传统套路一致，我们可以对预测分数设置一个阀值，只有预测的置信度大于阀值的query objects，我们将其输出显
        # 示（画出bbox），可以看官方show给我们的notebook。
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # 若指定要计算Decoder每层预测输出对应的loss，则记录对应的输出结果
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        # 类别数，不包含背景
        self.num_classes = num_classes
        # 对预测与GT进行匹配的算法
        self.matcher = matcher
        # 各种loss对应的权重
        self.weight_dict = weight_dict
        # 针对背景分类的loss权重
        self.eos_coef = eos_coef
        # 指定需要计算哪些loss（'labels','boxes','cardinality','masks'）
        self.losses = losses
        # 设置在分类loss中，前景的权重为1，背景权重由传进来的参数指定
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # 将这部分注册到buffer，能够被state_dict记录同时不会有梯度传播到此处
        self.register_buffer('empty_weight', empty_weight)

    """
    分类loss
    doc string里写的是NLL Loss，但实际调用的是CE Loss，这是因为在Pytorch实现中，CELoss实质上就是
    将LogSoftmax操作和NLL Loss封装在了一起，如果直接使用NLL Loss，那么需要先对预测结果作LogSoftmax操
    作，而使用CELoss则直接免去了这一步。
    """
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        # (b,num_queries=100, num_classes+1)
        # 分类预测结果
        src_logits = outputs['pred_logits']

        # 要理解_get_src_permutation_idx()在做什么。输入参数indices是匹配的预测结果与GT的
        # 索引，其形式在forward向前传播的indices = self.matcher(....)中已有说明。该方法返回一个tuple，
        # 代表所有匹配的预测结果的batch index（在当前batch中属于第几张图像）和 query index（图像中的第几个query对象）。
        # 这个tuple，第一个元素是各个object的batch index，第二个元素是各个object的query index，
        # shape都是（num_matched_queries1+num_matched_queries2+...，）
        idx = self._get_src_permutation_idx(indices)
        # 类似地，我们可以获得当前batch中所有匹配的GT所属的类别（target_classes_o），然后通过src_logits、target_classes_o
        # 就可以设置预测结果对应的GT了，这就是下面的target_classes。target_classes的shape和src_logits一致，代表每个
        # query objects对应的GT，首先将它们全部初始化为背景，然后根据匹配的索引（idx）设置匹配的GT（target_classes_p）类别。
        # 匹配的GT，（num_matched_queries1+num_matched_queries2+...）
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # （b，num_queries=100），初始化为背景
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # 匹配的预测索引对应的值置为匹配的GT
        target_classes[idx] = target_classes_o

        # “热身活动”做完后，终于可以开始计算loss了，注意在使用Pytorch的交叉熵时，需要将预测类别的那个维度转换到通道这个维度上（dim1）。
        # src_logits的shape变为（b,num_classes+1,num_queries=100）
        # 因为CELoss需要第一维对应类别数
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # class_error计算的是Top-1精度（百分数），即预测概率最大的那个类别与对应被分配的GT类别是否一致，这部分仅用于log，并不参与模型训练。
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    # 回归loss的计算包括预测框与GT的中心点和宽高的L1 loss以及GIoU loss
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        # (batch indices, query indices)
        # shape都是（num_matched_queries1+num_matched_queries2+...）
        idx = self._get_src_permutation_idx(indices)
        # outputs['pred_boxes']的shape是（b,num_queries=100,4）
        # src_boxes的shape是（num_matched_queries1+num_matched_queries2+...,4）
        src_boxes = outputs['pred_boxes'][idx]
        # （num_matched_objs1+num_matched_objs2+...,4）
        # num_matched_queries1+num_matched_queries2+..., 和 num_matched_objs1+num_matched_objs2+...
        # 是相等的，在forward部分的matcher的返回结果注释中有说明。
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        # 以下就是loss的计算。注意下 reduction 参数，若不显式进行设置，在Pytorch的实现中默认是'mean'，即返回所有涉及误差计算的元素的均值。
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        # num_boxes是一个batch图像中目标物体的数量
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # 由于generalized_box_iou返回的是每个预测结果与每个GT的giou，因此取对角线代表获取的是相互匹配的预测结果与GT的giou。
        # 在计算GIoU loss时，使用了torch.diag()获取对角线元素，这是因为generalized_box_iou()方法返回
        # 的是所有预测框与所有GT的GIoU，比如预测框有N个，GT有M个，那么返回结果就是NxM个GIoU。我们预先对
        # 匹配的预测框和GT进行了排列，即N个预测框中的第1个匹配M个GT中的第1个，N中第2个匹配M中第2个，..，N中
        # 第i个匹配M中第i个，于是我们要取相互匹配的那一项来计算loss。
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # （num_matched_queries1+num_matched_queries2+...）
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # （num_matched_queries1+num_matched_queries2+...）
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    # get_loss方法中并不涉及具体loss的计算，其仅仅是将不同类型的loss计算映射到对应的方法，最后将计算结果返回。
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # 接下来看下前向过程，了解下loss的计算过程。这里一定要先搞清楚模型输出（outputs）和GT（targets）的形式，对于outputs可参考下列的注释；
        # 而targets是一个包含多个dict的list，长度与batch size相等，其中每个dict的形式如同COCO数据集的标注，
        # outputs是DETR模型的输出，是一个dict，形式如下：
        # {'pred_logits':(b, num_queries=100, num_classes),
        # 'pred_boxes':(b, num_queries=100, 4),
        # 'aux_outputs':[{'pred_logits':..,'pred_boxes':...}, {...}, ...]}
        # 过滤掉中间层的输出，只保留最后一层的预测结果
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # 计算loss的一个关键前置步骤就是将模型输出的预测结果与GT进行匹配，对应下面self.matcher()部分，返回的indices的形式在下面注释中说明。
        # 将预测结果与GT匹配，indices是一个包含多个元组的list，长度与batch size相等，每个元组为（index_i，index_j），前者是匹配的预测索引，
        # 后者是GT索引，并且len(index_i)=len(index_j)=min(num_queries,num_targets_in_image)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        # 计算这个batch的图像中目标物体的数量，在所有分布式节点之间同步
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        # 接下来是计算各种类型的loss，并将对应结果存到一个dict中（下面的losses变量），self.get_loss()方法返回loss计算结果。
        losses = {}
        for loss in self.losses:
            # 计算特定类型的loss（这里的loss变量是字符串：'labels'，'boxes'，'cardinality','masks'，表示loss类型），
            # get_loss方法中并不涉及具体loss的计算，其仅仅是将不同类型的loss计算映射到对应的方法，最后将计算结果返回。
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        # 若模型输出包含了中间层输出，则一并计算对应的loss
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

# DETR的输出并不是最终预测结果的形式，还需要进行简单的后处理。但是这里的后处理并不是NMS哦！DETR预测的
# 是集合，并且在训练过程中经过匈牙利算法与GT一对一匹配学习，因此不存在重复框的情况。如doc string所述，
# 这个后处理指的是将DETR的输出转换为coco api对应的格式。以下需要关注的点有两个，一个是由于coco api中的
# 评估不包含背景类，于是这里在生成预测结果时直接排除了背景类；另一个是模型在回归部分的输出是归一化的值，
# 需要根据图像尺寸来还原。
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # (b, num_queries=100, num_classes+1)
        prob = F.softmax(out_logits, -1)
        # (b, num_queries=100),(b, num_queries=100)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        # (b,),(b,)
        img_h, img_w = target_sizes.unbind(1)
        # (b,4)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # (b, num_queries=100, 4)*(b,1,4)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            # 最后一层不适用ReLU函数激活
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    # 将预测结果和GT进行匹配的算法（这里使用的是匈牙利算法）
    # 由于 DETR 是预测结果是集合的形式，因此在计算loss的时候有个关键的前置步骤就是将预测结果和GT进行匹配，
    # 这里的GT是不包括背景的，未被匹配的预测结果就自动被归类为背景。匹配使用的是匈牙利算法，该算法主要用于
    # 解决与二分图匹配相关的问题，匈牙利算法参考：https://links.jianshu.com/go?to=https%3A%2F%2Fzhuanlan.zhihu.com%2Fp%2F96229700
    matcher = build_matcher(args)
    # 各类型loss的权重（分类使用的是交叉熵Loss，而回归loss包括了bbox的 L1 Loss（计算x、y、w、h的绝对值误差）与 GIoU Loss。）
    weight_dict = {'loss_ce': 1, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    # 若设置了masks参数，则代表分割任务，那么还需加入对应的loss类型。
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    # 若设置了aux_loss，即代表需要计算中间层预测结果对应的loss，那么也要设置对应的loss权重。
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            # 为中间层输出的loss也加上对应权重
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # 指定计算哪些类型的loss
    # 其中cardinality是计算预测为前景的数量与GT数量的L1误差，仅用作log展示，并不是真正的loss，不涉及反向传播梯度
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    # eos_coef用于在计算分类loss中前景和背景的相对权重
    # loss函数是通过实例化SetCriterion对象来构建。
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)
    # PostProcess 生成预测结果
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
