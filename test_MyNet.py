import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import time
from PIL import Image
from model.MyNet_EB1 import MyNet
from data import test_dataset
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# 设置CUDA设备
# torch.cuda.set_device(0)
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 解析输入参数
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
opt = parser.parse_args()

dataset_path = 'D:/object/my code2/Data/ESD/'

# 加载模型
model = MyNet()
model.load_state_dict(torch.load('./trained_models/ESD/'))

model.cuda()
model.eval()

test_datasets = ['test']

# 数据预处理和后处理
def preprocess_image(image, gt):
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    return image.cuda(), gt


def postprocess_output(res, gt_shape):
    res = F.interpolate(res, size=gt_shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    return (res * 255).astype(np.uint8)


for dataset in test_datasets:
    save_path = './results/ESD/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/imgs/'
    gt_root = dataset_path + dataset + '/masks/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        image, gt = preprocess_image(image, gt)
        time_start = time.time()
        res, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, s5, s5_sig = model(image)

        time_end = time.time()
        time_sum += (time_end - time_start)

        res_image = postprocess_output(res, gt.shape[-2:])
        Image.fromarray(res_image).save(save_path + name)

        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))

