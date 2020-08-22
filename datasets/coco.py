# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T

# CocoDetection 这个类继承了torchvision.datasets.CocoDetection。
class CocoDetection(torchvision.datasets.CocoDetection):
    # 在类的初始化方法中，首先调用父类的初始化方法，将图像文件及标注文件的路径传进去。transforms 是用于数据增强的方法；根据名字
    # 来看，ConvertCocoPolysToMask() 这个对象是将数据标注的多边形坐标转换为掩码，但其实不仅仅是这样，或者说不一定是这样，因为
    # 需要根据传进去的参数 return_masks 来确定，若传进来的 return_masks 值不为True，那么实质上是没有将数据标注的多边形坐标转换
    # 为掩码的。另外，需要提下COCO数据集中标注字段annotation的格式，对于目标检测任务，其格式如下：
    # "annotations":
    #     [
    #         {
    #         　　"segmentation": [[37.31,373.02,57.4,216.61,67.44,159.21,77.49,113.29,91.84,86.03,123.41,84.59,162.15,96.07,215.25,86.03,261.17,70.24,285.56,68.81,337.22,68.81,411.84,93.2,454.89,107.55,496.5,255.35,513.72,262.53,552.47,292.66,586.0,324.23,586.0,381.63,586.0,449.08,586.0,453.38,578.3,616.97,518.03,621.27,444.84,624.14,340.09,625.58,136.32,625.58,1.43,632.75,7.17,555.26,5.74,414.64]],
    #          　 "area": 275709.8110500001,
    #             "iscrowd": 0,
    #             "image_id": 285,
    #             "bbox": [1.43,68.81,584.57,563.94],
    #             "category_id": 23,
    #             "id": 587562
    #         }
    #     ]
    # segmentation格式取决于这个实例是一个单个的对象（即iscrowd=0，将使用polygons格式）还是一组对象（即iscrowd=1，将使用RLE格式）
    # 当 "iscrowd" 字段为0时，segmentation就是polygon的形式，比如这时的 "segmentation" 的值可能
    # 为 [[510.66, 423.01, 511.72, 420.03, 510.45......], ..]，其中是一个个polygon即多边形，这
    # 些数按序两两组成多边形各个点的横、纵坐标，也就是说，表示polygon的list中如果有n个数（必定是偶
    # 数），那么就代表了 n/2 个点坐标。
    # 每个对象（不管是iscrowd=0还是iscrowd=1）都会有一个矩形框bbox ，矩形框左上角的坐标和矩形框的长宽会以数组的形式提供，数组第一个元素就是左上角的横坐标值。
    # 至于取数据用到的 __getitem__ 方法，首先也是调用父类的这个初始化方法获得图像和对应的标签，然后 prepare 就是调用 ConvertCocoPolysToMask()
    # 这个对象对图像和标签进行处理，之后若有指定数据增强，则进一步进行对应的处理，最后返回这一系列处理后的图像和对应的标签。
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        # ConvertCocoPolysToMask() 仅在传入的参数 return_masks 为True时做了
        # 将多边形转换为掩码的操作，该对象的主要工作其实是过滤掉标注为一组对象的
        # 数据，以及筛选掉bbox坐标不合法的那批数据。现在我们来看看 convert_coco_poly_to_mask()
        # 这个方法即将多边形坐标转换为掩码是如何操作的。(详细解释和代码分析：https://www.jianshu.com/p/c5e95a80fd98)
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        # 在原有COCO的标注基础上，重新定制下标注
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            # 数据增强
            img, target = self._transforms(img, target)
        return img, target


"""
该方法中调用的 frPyObjects 和 decode 都是 coco api（pycocotools）中的方法，将每个多边形
结合图像尺寸解码为掩码，然后将掩码增加至3维（若之前不足3维）。这里有个实现上的细节——为何要加
一维呢？因为我们希望的是这个mask能够在图像尺寸范围（h, w）中指示每个点为0或1，在解码后，mask
的shape应该是 (h,w)，加一维变为 (h,w,1)，然后在最后一个维度使用any()后才能维持原来的维度即(h,w)；
如果直接在(h,w)的最后一维使用any()，那么得到的shape会是(h,)，各位可以码码试试。最后，将一个
个多边形转换得到的掩码添加至列表，堆叠起来形成张量后返回。
"""
def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


# ConvertCocoPolysToMask() 仅在传入的参数 return_masks 为True时做了将多边形转换为掩码的操作，该对象的主要工作其实是过滤掉标注为一组
# 对象的数据，以及筛选掉bbox坐标不合法的那批数据。
# 这里的 target 是一个list，其中包含了多个字典类型的annotation，每个annotation的格式如上面21行注释中所示。这里
# 将 "iscrowd" 为1的数据（即一组对象，如一群人）过滤掉了，仅保留标注为单个对象的数据。另外这里对bbox的形式做了转
# 换，将"xywh"转换为"x1y1x2y2"的形式，并且将它们控制图像尺寸范围内。
class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        # 为boxes生成tensor，并转换成4列（一项里面有四个）
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # 至此boxes形如：
        # [[x0,y0,w0,h0],
        # [x1,y1,w1,h1],
        # ...,
        # [xn,yn,wn,hn],
        # ]
        # 计算bbox右下角坐标(把每一行的最后两列，分别加到每一行的前两列)
        boxes[:, 2:] += boxes[:, :2]
        # 至此boxes形如：
        # [[x00,y00,x01,y01],
        # [x10,y10,x11,y11],
        # ...,
        # [xn0,yn0,xn1,yn1],
        # ]
        # 切片操作[开始：结束：步长]，双冒号仅仅是没有中间的结束而已
        # 将bbox的左上角和右下角坐标控制在图像尺寸范围内
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            # 根据标注的segmentation字段生成掩码，用与分割任务
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)
        # keep 代表那些有效的bbox，即左上角坐标小于右下角坐标那些，过滤掉无效的那批
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # 在进行完处理和过滤操作后，更新annotation里各个字段的值
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # 新增 "orig_size" 和 "size" 两个 key
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        # 最后返回处理后的图像和标签。
        return image, target

# DETR 的整体工作很solid，没有使用骚里骚气的数据增强，那么我们就来看看它究竟在数据增强方面做了啥
# 可以看到，真的是很“老土”！就是归一化、随机反转、缩放、裁剪，除此之外，没有了，可谓大道至简
def make_coco_transforms(image_set):
    # T 是项目中的datatsets/transforms.py模块，以上各个数据增强的方法在该模
    # 块中的实现和 torchvision.transforms 中的差不多，其中ToTensor()会
    # 将图像的通道维度排列在第一个维度，并且像素值归一化到0-1范围内；而Normalize()
    # 则会根据指定的均值和标准差对图像进行归一化，同时将标签的bbox转换为Cx Cy W H形式，后归一化到0-1，此处不再进行解析，感兴趣的可以去参考源码。
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

# 通常，很多项目在数据处理部分都会相对复杂，一方面固然是因为数据处理好了模型才能进行有效训练与学习，而另一方
# 面则是为了适应任务需求而“不得已”处理成这样，其中还可能会使用到一些算法技巧，但是在 DETR中，真的太简单了，
# coco api 几乎搞定了一切，然后搞几个超级老土的 data augmentation，完事，666！
def build(image_set, args):
    root = Path(args.coco_path)
    # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
    # 检查数据文件路径的有效性
    assert root.exists(), f'provided COCO path {root} does not exist'
    # 对于目标检测，COCO数据集的标注文件是'path_to_coco/annotation/instance_xxx2017.jason'
    mode = 'instances'
    # 构造一个字典类型的 PATHS 变量来映射训练集与验证集的路径
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }
    # image_set指代训练集/验证集
    img_folder, ann_file = PATHS[image_set]
    # 使用COCO API来构造数据集
    # 实例化一个 CocoDetection() 对象,以及应用数据增强make_coco_transforms
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
