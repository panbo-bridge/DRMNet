import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.utils.data as data
import os
import warnings
from time import time
from networks.unet import Unet
from networks.dunet import Dunet
from networks.linknet import LinkNet34
from networks.dlinknet import DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool
from networks.nllinknet import NL34_LinkNet
from modeling.deeplab import DeepLab
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder
from tqdm import tqdm
import numpy as np
from eval import IOUMetric
import random
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU. 
    np.random.seed(seed) # Numpy module. 
    random.seed(seed) # Python random module. 
    torch.manual_seed(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
    #torch.use_deterministic_algorithms(True)
set_seed()
warnings.filterwarnings("ignore")
dataset_name = "mass"
model_name = "D-linknet"
if dataset_name == "deepglobe":
    ROOT = 'dataset/train/'
    NAME = f'{dataset_name}_trainlog_{model_name}'  # 保存日志名
    imagelist = filter(lambda x: x.find('sat') != -1, os.listdir(ROOT))
    alllist = list(map(lambda x: x[:-8], imagelist))
    train_len = int(len(alllist) * 0.8)
    trainlist = alllist[:train_len]
    vallist = alllist[train_len:]
    train_root = ROOT
    val_root = ROOT
else:
    NAME = f'{dataset_name}_trainlog_{model_name}'  # 保存日志名
    train_root = "./mass_road/train/map"
    val_root = "./mass_road/valid/map"
    trainlist = os.listdir(train_root)
    vallist = os.listdir(val_root)

batchsize = 32

if __name__ == '__main__':
    solver = MyFrame(DinkNet34, dice_bce_loss, 2e-4)
    dataset = ImageFolder(trainlist, train_root, train=True,dataset_name=dataset_name)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=16)
    val_dataset = ImageFolder(vallist, val_root,dataset_name=dataset_name)
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize,
        shuffle=False,
        num_workers=16)
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()
    no_optim = 0
    total_epoch = 300  # 训练轮数
    train_epoch_best_loss = 100.
    best_miou = 0
    best_miou_epoch = 0
    for epoch in range(1, total_epoch + 1):
        labels = []
        predicts = []
        train_epoch_loss = 0
        for img, mask, erased_gt in tqdm(iter(data_loader)):
            solver.set_input(img, mask, erased_gt)
            train_loss = solver.optimize()
            train_epoch_loss += train_loss
        train_epoch_loss /= len(iter(data_loader))
        print('********', file=mylog)
        print('epoch:', epoch, 'time:', int(time() - tic), file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)

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
        #acc, acc_cls, iou, miou, fwavacc = el.evaluate(predicts, labels)
        miou = el.evaluate(predicts, labels)
        # print('acc: ', acc, file=mylog)
        # print('acc_cls: ', acc_cls, file=mylog)
        # print('iou: ', iou, file=mylog)
        # print('miou: ', miou, file=mylog)
        # print('fwavacc: ', fwavacc, file=mylog)
        # print('acc: ', acc,'iou: ', iou,'miou: ', miou)
        if miou > best_miou:
            best_miou = miou
            best_miou_epoch = epoch
            solver.save('weights/' + NAME + '.pt')
        print("best miou: ",best_miou,"best_miou_epoch",best_miou_epoch, file=mylog)
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            #solver.save('weights/' + NAME + '.pt')
        if no_optim > 6:
            print('early stop at %d epoch' % epoch, file=mylog)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break
            solver.load('weights/' + NAME + '.pt')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()
