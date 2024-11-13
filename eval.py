# -- coding: utf-8 --
"""
@Time：2023-05-19 21:51
@Author：zstar
@File：eval.py
@Describe：用于评估分割效果相关指标，本示例仅做单图评估，更多图片可类似拓展
"""
import os
import cv2
import numpy as np
import torch
import warnings
from torch.autograd import Variable as V
from framework import MyFrame
from loss import dice_bce_loss
from networks.dlinknet import DinkNet34
from networks.linknet import LinkNet34
from networks.unet import Unet
warnings.filterwarnings("ignore")
from data import ImageFolder
from tqdm import tqdm
from modeling.deeplab import DeepLab
class IOUMetric_:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        # miou
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        # mean acc
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return acc, acc_cls, iou, miou, fwavacc


class IOUMetric:
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def evaluate(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            assert len(lp.flatten()) == len(lt.flatten())
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        
        # IoU and mIoU
        iou = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        miou = np.nanmean(iou)
        
        # Overall accuracy
        acc = np.diag(self.hist).sum() / self.hist.sum()
        
        # Mean accuracy per class
        acc_cls = np.nanmean(np.diag(self.hist) / self.hist.sum(axis=1))
        
        # Frequency-weighted IoU
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()

        # Recall per class
        recall = np.diag(self.hist) / self.hist.sum(axis=1)
        
        # Precision per class
        precision = np.diag(self.hist) / self.hist.sum(axis=0)
        
        # F1 Score per class
        f1 = 2 * (precision * recall) / (precision + recall)
        
        # Mean Recall and Mean F1 Score
        mean_recall = np.nanmean(recall)
        mean_f1 = np.nanmean(f1)

        # print("Overall Accuracy": acc,"Mean Accuracy per Class": acc_cls, "IoU per Class": iou, "Mean IoU": miou,\
        #     "Frequency Weighted IoU": fwavacc, \
        #     "Recall per Class": recall, \
        #     "Mean Recall": mean_recall, \
        #     "Precision per Class": precision, \
        #     "F1 Score per Class": f1, \
        #     "Mean F1 Score": mean_f1 \
        # )
        
        print("Precision",precision)
        print("recall",recall)
        print("f1",f1)
        print("miou",miou)
        return miou
def test():
    labels = []
    predicts = []
    ROOT = 'dataset/train/'
    NAME = 'trainlog_linkNet34'  # 保存日志名
    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
    alllist = list(map(lambda x: x[:-8], imagelist))
    train_len = int(len(alllist) * 0.8)
    vallist = alllist[train_len:]
    val_dataset = ImageFolder(vallist, ROOT)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=16)
    solver = MyFrame(DeepLab, dice_bce_loss, 2e-4)
    solver.load("/opt/data/private/workspace/Road-Extraction-master/weights/trainlog_deeplabv3.pt")
    for img, mask, erased_gt in tqdm(iter(val_data_loader)):
        solver.set_input(img, mask, erased_gt)
        pred, _ = solver.test_batch()
        for i in range(pred.shape[0]):
            mask_i = mask[i].squeeze(0)
            pred_i = pred[i]
            pred_i = np.array(pred_i, np.int64)
            mask_i = np.array(mask_i, np.int64)

            labels.append(mask_i)
            predicts.append(pred_i)
    el = IOUMetric()
    el.evaluate(predicts, labels)
    # acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
    # print('acc: ', acc)
    # print('acc_cls: ')
    # print('iou: ', iou)
    # print('miou: ', miou)
    # print('fwavacc: ', fwavacc)
def test_one_image():
    labels = []
    predicts = []
    if 0:
        data_root = "./dataset/train"
        data_root_ = "./dataset/vision"
        imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(data_root))
        alllist = list(map(lambda x: x[:-8], imagelist))
        train_len = int(len(alllist) * 0.8)
        trainlist = alllist[:train_len]
        vallist = alllist[train_len:]
        # 加载模型
        solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
        solver.load("/opt/data/private/workspace/Road-Extraction-master/weights/trainlog_dinkNet34.pt")
        vision = True
        for name in tqdm(vallist):
            img_path = f"{data_root}/{name}_sat.jpg"
            label_path = f"{data_root}/{name}_mask.png"
            pred_path = f"{data_root}/{name}_pred.png"
            if vision:
                save_img_path = f"{data_root_}/{name}_sat.jpg"
                save_label_path = f"{data_root_}/{name}_mask.png"
                save_pred_path = f"{data_root_}/{name}_pred.png"
            # 读取图片，分割
            img_size = (512, 512)
            img = cv2.imread(img_path)
            if vision:
                cv2.imwrite(save_img_path, img)
            
            img = cv2.resize(img,img_size)
            img = img[None, ...].transpose(0, 3, 1, 2)
            img = V(torch.Tensor(np.array(img, np.float32) / 255.0 * 3.2 - 1.6).cuda())
            predict = solver.test_one_img(img)
            predict = np.array(predict, np.int64)
            predict_img = predict.astype(np.uint8) * 255
            cv2.imwrite(pred_path, predict_img)
            if vision:
                cv2.imwrite(save_pred_path, predict_img)
            # 读取label，二值化处理
            label = cv2.imread(label_path, 0)
            if vision:
                cv2.imwrite(save_label_path, label)
            label = cv2.resize(label, img_size)
            #label[label > 0] = 1
            label[label >= 0.5] = 1
            label[label <= 0.5] = 0
            # 添加进评估列表，更多图片同理
            predicts.append(predict)
            labels.append(label)

        # 评估
        el = IOUMetric()
        el.evaluate(predicts, labels)
        # acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
        # print('acc: ', acc)
        # print('acc_cls: ', acc_cls)
        # print('iou: ', iou)
        # print('miou: ', miou)
        # print('fwavacc: ', fwavacc)
    else:
        train_root = "./mass_road/train/map"
        val_root = "./mass_road/valid/map"
        # 加载模型
        solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
        solver.load("/opt/data/private/workspace/Road-Extraction-master/weights/mass_trainlog_D-linknet.pt")
        for data_root in [train_root, val_root]:
        #for data_root in [val_root]:
            alllist = os.listdir(data_root)
            for name in tqdm(alllist):
                label_path = f"{data_root}/{name}"
                img_path = label_path.replace("map","sat").replace("tif","tiff")
                pred_path = f"{os.path.dirname(data_root)}/pred/{name}"
                # 读取图片，分割
                img_size = (256, 256)
                img = cv2.imread(img_path)
                img = cv2.resize(img,img_size)
                img = img[None, ...].transpose(0, 3, 1, 2)
                img = V(torch.Tensor(np.array(img, np.float32) / 255.0 * 3.2 - 1.6).cuda())
                predict = solver.test_one_img(img)
                predict = np.array(predict, np.int64)
                predict_img = predict.astype(np.uint8) * 255
                cv2.imwrite(pred_path, predict_img)
                # 读取label，二值化处理
                label = cv2.imread(label_path, 0)
                label = cv2.resize(label, img_size)
                #label[label > 0] = 1
                label[label >= 0.5] = 1
                label[label <= 0.5] = 0
                # 添加进评估列表，更多图片同理
                predicts.append(predict)
                labels.append(label)

            # 评估
            el = IOUMetric()
            el.evaluate(predicts, labels)
            # acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
            # print('acc: ', acc)
            # print('acc_cls: ', acc_cls)
            # print('iou: ', iou)
            # print('miou: ', miou)
            # print('fwavacc: ', fwavacc)
if __name__ == "__main__":
    test_one_image()