# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

# 首先将模型设置为训练模式，这样梯度才能进行反向传播，从而更新模型参数的权重。
# 注意到这里同时将 criterion 对象也设为train模式，它是 SetCriterion 类的一
# 个对象实例，代表loss函数，看了下相关代码发现里面并没有需要学习的参数，因此
# 感觉之类可以将这行代码去掉，后面我会亲自实践看看，朋友们也可一试。

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    # 这里用到了一个类 MetricLogger（位于 detr/util/misc.py），它主要用于log输出，
    # 其中使用了一个defaultdict来记录各种数据的历史值，这些数据为 SmoothValue（位
    # 于 detr/util/misc.py） 类型，该类型通过指定的窗口大小（window_size）
    # 来存储数据的历史步长（比如1就代表不存储历史记录，每次新的值都会覆盖旧的），并且
    # 可以格式化输出。另外 SmoothValue 还实现了统计中位数、均值等方法，并且能够在各
    # 进程间同步数据。MetricLogger 除了通过key来存储SmoothValue以外，最重要的就是其
    # 实现了一个log_every的方法，这个方法是一个生成器，用于将每个batch的数据取出（yeild），
    # 然后该方法内部会暂停在此处，待模型训练完一次迭代后再执行剩下的内容，进行各项统计，
    # 然后再yeild下一个batch的数据，暂停在那里，以此重复，直至所有batch都训练完。这种方
    # 式在其它项目中比较少见，感兴趣的炼丹者们可以一试，找些新鲜感
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        # 各部分loss（如分类loss、回归loss等）的加权和
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        # 在计算出loss后，若采用了分布式训练，那么就在各个进程间进行同步，默认是总和/进程数量
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        # 若梯度溢出了，那么此时会产生梯度爆炸，于是就直接结束训练。
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # 为避免梯度爆炸，在训练过程中，对梯度进行裁剪，裁剪方式有很
        # 多种，可以直接对梯度值处理，这里的方式是对梯度的范式做截断，
        # 默认是第二范式，即所有参数的梯度平方和开方后与一个指定的最
        # 大值（下图中max_norm）相比，若比起大，则按比例对所有参数的
        # 梯度进行缩放。
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            # 对梯度的范式进行截断，默认是第二范式
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    # 将 MetricLogger 统计的各项数据在进程间进行同步，同时返回它们的历史均值
    # 这个历史均值：global_avg是SmoothedValue类里的属性方法（用@property修饰），返回的是
    # 在各个进程同步后的历史均值。例如对于loss这项数据，在训练过程中被计算了n次，那么历史均
    # 值就是这n次的总和在进程间同步除以n
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # 关于 MetricLogger 和 SmoothValue 的具体实现这里就不作解析了，这只是作者的个人喜好，
    # 用于训练过程中数据的记录与展示，和模型的工作原理及具体实现无关，大家如果想要将 DETR
    # 用到自己的项目上，完全可以不care这部分。对于 MetricLogger 和 SmoothValue 的这种做
    # 法，我们可以学习下里面的技巧，抽象地继承，而不必生搬硬套。


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator
