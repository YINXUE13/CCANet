import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import os, argparse, logging
from datetime import datetime
from model.MyNet_EB1 import MyNet  # 请确保你的 MyNet19 模型路径正确
from data import get_loader
from utils.utils import clip_gradient, adjust_lr, AvgMeter
import pytorch_iou
import pytorch_fm
import random
from torch.backends import cudnn
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

cudnn.benchmark = True
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)
random.seed(2025)
np.random.seed(2025)


def train(train_loader, model, optimizer, epoch, total_step, opt):
    model.train()
    loss_record1, loss_record2, loss_record3, loss_record4, loss_record5 = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts = pack
        images = Variable(images).cuda()
        gts = Variable(gts).cuda()

        # forward + backward + optimize
        s1, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, s5, s5_sig = model(images)
        loss1 = nn.BCEWithLogitsLoss()(s1, gts) + pytorch_iou.IOU(size_average=True)(s1_sig, gts) + pytorch_fm.FLoss()(s1_sig, gts)
        loss2 = nn.BCEWithLogitsLoss()(s2, gts) + pytorch_iou.IOU(size_average=True)(s2_sig, gts) + pytorch_fm.FLoss()(s2_sig, gts)
        loss3 = nn.BCEWithLogitsLoss()(s3, gts) + pytorch_iou.IOU(size_average=True)(s3_sig, gts) + pytorch_fm.FLoss()(s3_sig, gts)
        loss4 = nn.BCEWithLogitsLoss()(s4, gts) + pytorch_iou.IOU(size_average=True)(s4_sig, gts) + pytorch_fm.FLoss()(s4_sig, gts)
        loss5 = nn.BCEWithLogitsLoss()(s5, gts) + pytorch_iou.IOU(size_average=True)(s5_sig, gts) + pytorch_fm.FLoss()(s5_sig, gts)
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss_record1.update(loss1.data, opt.batchsize)
        loss_record2.update(loss2.data, opt.batchsize)
        loss_record3.update(loss3.data, opt.batchsize)
        loss_record4.update(loss4.data, opt.batchsize)
        loss_record5.update(loss5.data, opt.batchsize)

        if i % 20 == 0 or i == total_step:
            logging.info('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}'.
                         format(datetime.now(), epoch, opt.epoch, i, total_step,
                                opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                                loss2.data))

    # 模型保存
    save_path = './trained_models_352/ORSSD-1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if epoch % 5 == 0 or epoch >= 40:
        torch.save(model.state_dict(), os.path.join(save_path, f'MYNet.{epoch}.pth'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=60, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
    opt = parser.parse_args()

    logging.info(f'Learning Rate: {opt.lr}')

    # build models
    model = MyNet()
    #model.load_state_dict(torch.load(r'D:\object\my code2\MYNet-master-62\MyNet-main\trained_models_352\AUG\MYNet.10.pth'))
    model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, opt.lr)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    image_root = 'D:/object/my code2/Dataother/ORSSD/train/imgs/'
    gt_root = 'D:/object/my code2/Dataother/ORSSD/train/masks/'
    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
    total_step = len(train_loader)

    logging.info("Training started!")
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model, optimizer, epoch, total_step, opt)


if __name__ == '__main__':
    main()



