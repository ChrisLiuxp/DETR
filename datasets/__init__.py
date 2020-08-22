# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

# 方法内部根据用户参数来构造用于目标检测/全景分割的数据集。image_set 是一个字符类型的参数，代表要构造的是训练集还是验证集。
def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        # 针对目标检测任务，我们来看看 build_coco() 这个方法的内容，该方法位于datasets/coco.py。
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
