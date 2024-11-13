"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import torch
import torch.utils.data as data

import cv2
import numpy as np
import os


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        # image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask
def rgbm_default_loader(id, root, train,dataset_name="deepglobe"):
    if dataset_name == "deepglobe":
        img = cv2.imread(os.path.join(root, '{}_sat.jpg').format(id))
        mask = cv2.imread(os.path.join(root + '{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(root + '{}_pred.png').format(id), cv2.IMREAD_GRAYSCALE)
        train_imgsize = (512, 512)
        val_imgsize = (512, 512)

    else:
        mask = cv2.imread(os.path.join(root, id), cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(os.path.join(root, id).replace("map","pred"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(os.path.join(root, id).replace("map","sat").replace("tif","tiff"))
        train_imgsize = (256, 256)
        val_imgsize = (256, 256)


    if train:
        img = cv2.resize(img, train_imgsize, interpolation=cv2.INTER_LINEAR,)
        mask = cv2.resize(mask, train_imgsize, interpolation=cv2.INTER_LINEAR,)
        pred = cv2.resize(pred, train_imgsize, interpolation=cv2.INTER_LINEAR,)
    else:
        img = cv2.resize(img, val_imgsize, interpolation=cv2.INTER_LINEAR,)
        mask = cv2.resize(mask, val_imgsize, interpolation=cv2.INTER_LINEAR,)     
        pred = cv2.resize(pred, val_imgsize, interpolation=cv2.INTER_LINEAR,)

    mask = np.expand_dims(mask, axis=2)
    pred = np.expand_dims(pred, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    pred = np.array(pred, np.float32).transpose(2, 0, 1) / 255.0
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, pred, mask

def default_loader(id, root, train, dataset_name="deepglobe"):
    if dataset_name == "deepglobe":
        img = cv2.imread(os.path.join(root, '{}_sat.jpg').format(id))
        mask = cv2.imread(os.path.join(root + '{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
        train_imgsize = (512, 512)
        val_imgsize = (512, 512)
    else:
        mask = cv2.imread(os.path.join(root, id), cv2.IMREAD_GRAYSCALE)
        #mask = cv2.imread(os.path.join(root + '{}.tif').format(id), cv2.IMREAD_GRAYSCALE) 
        img = cv2.imread(os.path.join(root, id).replace("map","sat").replace("tif","tiff"))
        if img is None or mask is None:
            print(id)
        train_imgsize = (256, 256)
        val_imgsize = (256, 256)
    if train:
        img = cv2.resize(img, train_imgsize, interpolation=cv2.INTER_LINEAR,)
        mask = cv2.resize(mask, train_imgsize, interpolation=cv2.INTER_LINEAR,)
    else:
        img = cv2.resize(img, val_imgsize, interpolation=cv2.INTER_LINEAR,)
        mask = cv2.resize(mask, val_imgsize, interpolation=cv2.INTER_LINEAR,)     
    if train:
        img = randomHueSaturationValue(img,
                                    hue_shift_limit=(-30, 30),
                                    sat_shift_limit=(-5, 5),
                                    val_shift_limit=(-15, 15))

        img, mask = randomShiftScaleRotate(img, mask,
                                        shift_limit=(-0.1, 0.1),
                                        scale_limit=(-0.1, 0.1),
                                        aspect_limit=(-0.1, 0.1),
                                        rotate_limit=(-0, 0))
        img, mask = randomHorizontalFlip(img, mask)
        img, mask = randomVerticleFlip(img, mask)
        img, mask = randomRotate90(img, mask)

    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2, 0, 1) / 255.0
    mask[mask >= 0.5] = 1
    mask[mask <= 0.5] = 0
    # mask = abs(mask-1)
    return img, mask


class ImageFolder(data.Dataset):

    def __init__(self, trainlist, root, train=False,refine=False,dataset_name="deepglobe"):
        self.ids = trainlist
        self.dataset_name = dataset_name
        self.loader = default_loader
        self.loader_rgbm = rgbm_default_loader
        self.root = root
        self.train = train
        self.refine = refine
        self.angle_theta = 10
    def getCorruptRoad(
        self, road_gt, height, width, artifacts_shape="linear", element_counts=8
    ):
        # False Negative Mask
        FNmask = np.ones((height, width), np.float32)
        # False Positive Mask
        FPmask = np.zeros((height, width), np.float32)
        indices = np.where(road_gt == 1)

        if artifacts_shape == "square":
            shapes = [[16, 16], [32, 32]]
            # FNmask
            if len(indices[0]) == 0:  # no road pixel in GT
                pass
            else:
                for c_ in range(element_counts):
                    c = np.random.choice(len(shapes), 1)[
                        0
                    ]  # choose random square size
                    shape_ = shapes[c]
                    ind = np.random.choice(len(indices[0]), 1)[
                        0
                    ]  # choose a random road pixel as center for the square
                    row = indices[0][ind]
                    col = indices[1][ind]

                    FNmask[
                        row - shape_[0] / 2: row + shape_[0] / 2,
                        col - shape_[1] / 2: col + shape_[1] / 2,
                    ] = 0
            # FPmask
            for c_ in range(element_counts):
                # choose random square size
                c = np.random.choice(len(shapes), 2)[0]
                shape_ = shapes[c]
                row = np.random.choice(height - shape_[0] - 1, 1)[
                    0
                ]  # choose random pixel
                col = np.random.choice(width - shape_[1] - 1, 1)[
                    0
                ]  # choose random pixel
                FPmask[
                    row - shape_[0] / 2: row + shape_[0] / 2,
                    col - shape_[1] / 2: col + shape_[1] / 2,
                ] = 1

        elif artifacts_shape == "linear":
            # FNmask
            if len(indices[0]) == 0:  # no road pixel in GT
                pass
            else:
                for c_ in range(element_counts):
                    c1 = np.random.choice(len(indices[0]), 1)[
                        0
                    ]  # choose random 2 road pixels to draw a line
                    c2 = np.random.choice(len(indices[0]), 1)[0]
                    cv2.line(
                        FNmask,
                        (indices[1][c1], indices[0][c1]),
                        (indices[1][c2], indices[0][c2]),
                        0,
                        self.angle_theta * 2,
                    )
            # FPmask
            for c_ in range(element_counts):
                row1 = np.random.choice(height, 1)
                col1 = np.random.choice(width, 1)
                row2, col2 = (
                    row1 + np.random.choice(50, 1),
                    col1 + np.random.choice(50, 1),
                )
                cv2.line(FPmask, (int(col1), int(row1)), (int(col2), int(row2)),
                         1, self.angle_theta * 2)

        erased_gt = (road_gt * FNmask) + FPmask
        erased_gt[erased_gt > 0] = 1

        return erased_gt
    def __getitem__(self, index):
        id = self.ids[index]
        if self.refine:
            img, mask, pred = self.loader_rgbm(id, self.root, self.train, self.dataset_name)
            pred = torch.Tensor(pred)
        else:
            img, mask = self.loader(id, self.root, self.train, self.dataset_name)
        c, h, w = img.shape
        # erased_gt = self.getCorruptRoad(mask, h, w) #pretrain refine model
        # erased_gt = torch.from_numpy(erased_gt)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        

        if self.refine:
            return img, mask, pred
        else:
            return img, mask, 0


    def __len__(self):
        # return len(self.ids)
        return len(list(self.ids))
