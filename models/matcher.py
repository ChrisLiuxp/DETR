# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


"""
如doc string所述，GT是不包含背景类的，通常预测集中的物体数量（默认为100）会比图像中实际存在的目标数量多，匈
牙利算法按1对1的方式进行匹配，没有被匹配到的预测物体就自动被归类为背景（non-objects）。
"""
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        # 以下cost_xx代表各类型loss的相对权重，在匈牙利算法中，描述为各种度量的相对权重会更合适，因此，这里命名使用的是'cost'
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    # 现在来看看前向过程，注意这里是不需要梯度的。
    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # 首先将预测结果和GT进行reshape，并对应起来，方便进行计算。
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        # (num_targets1+num_targets2+...,)
        tgt_ids = torch.cat([v["labels"] for v in targets])
        # (num_targets1+num_targets2+...,4)
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # 然后就可以对各种度量（各类型loss）进行计算。这里的cost与之前的分类和回归的loss并不完全一样，比如对于分类
        # 来说，loss计算使用的是交叉熵，而这里为了更加简便，直接采用1减去预测概率的形式，同时由于1是常数，于是作者
        # 甚至连1都省去了，有够机智（懒）的...
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # (batch_size*num_queries,num_targets1+num_targets2+...,)
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        # 另外，在计算bbox的L1误差时，使用了torch.cdist()，其中设置参数p=1代表L1范式（默认是p=2，即L2范式），这个方法会对每个预测
        # 框与GT都进行误差计算，如预测框有N个，GT有M个，结果就会有NxM个值。
        # out_bbox中的每个元组都与tgt_bbox中的计算l1 loss(p=1):|x-x`hat|+|y-y`hat|+|w-w`hat|+|h-h`hat|
        # (batch_size*num_queries, num_targets1+num_targets2+...,)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        # (batch_size*num_queries, num_targets1+num_targets2+...,)
        # 省略了常数1
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        # 接着对各部分度量加权求和，得到一个总度量
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # (batch_size, num_queries, num_targets1+num_targets2+...,)
        C = C.view(bs, num_queries, -1).cpu()

        # 统计了当前batch中每张图像的GT数量，这个操作是为什么呢？接着看，你会发现这招很妙！
        # num_targets1+num_targets2+...
        sizes = [len(v["boxes"]) for v in targets]
        # C.split()在最后一维按各张图像的目标数量进行分割，这样就可以在各图像中将预测结果与GT进行匹配了。
        # 匹配方法使用的是scipy优化模块中的linear_sum_assignment()，其输入是二分图的度量矩阵，该方法是计算这个二分图度量
        # 矩阵的最小权重分配方式，返回的是匹配方案对应的矩阵行索引和列索引。
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # for each image in a batch, len(i)=len(j)=min(num_queries, num_target_boxes)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


"""
build_matcher()方法返回HungarianMatcher()对象，其中实现了匈牙利算法，在这里用于预测集（prediction set）和GT的匹配，
最终匹配方案是选取“loss总和”最小的分配方式。注意CW对loss总和这几个字用了引号，因为其与loss函数中计算的loss并不完全一致，
但实际代表的意义是相同的，接下来我们看代码实现就会一清二楚。
吾以为，loss函数的设计是DL项目中最重要的部分之一，它本质是一个优化问题，关系到模型的学习，里面往往会涉及许多技巧，也是最能体
现作者思想的部分，CW每次看项目的源码时，最打起精神的就是这一part了。
"""
def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
