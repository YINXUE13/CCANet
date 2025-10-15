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

def visualize_feature_map(img_batch, img_name, num, save_path='feature_maps/', dpi=100,
                                     low_threshold=0.3, high_threshold=0.7, cmap_name='jet'):
    """
    支持下阈值映射蓝色、上阈值映射红色，中间正常渐变。

    参数:
    - low_threshold: float, 小于该值强制变蓝（数值映射为0）
    - high_threshold: float, 大于该值强制变红（数值映射为1）
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    feature_map = torch.squeeze(img_batch, 0).cpu()
    if len(feature_map.size()) == 2:
        feature_map = feature_map.unsqueeze(0)

    feature_map_sum = torch.sum(feature_map, dim=0)

    # 归一化
    feature_map_sum = (feature_map_sum - feature_map_sum.min()) / (feature_map_sum.max() - feature_map_sum.min())

    # 反转，使高值变红
    # feature_map_sum = 1.0 - feature_map_sum

    feature_map_sum = feature_map_sum.detach().numpy()

    # 三段映射：
    # < low_threshold => 0 (蓝)
    # > high_threshold => 1 (红)
    # 中间线性映射到0~1区间
    below_low = feature_map_sum < low_threshold
    above_high = feature_map_sum > high_threshold
    middle = (~below_low) & (~above_high)

    # 先全部置为0
    mapped = np.zeros_like(feature_map_sum)

    # 中间线性映射到0~1区间
    if high_threshold != low_threshold:
        mapped[middle] = (feature_map_sum[middle] - low_threshold) / (high_threshold - low_threshold)
    else:
        mapped[middle] = 0  # 避免除0

    # 低于下阈值的置0，映射为蓝
    mapped[below_low] = 0.0

    # 高于上阈值的置1，映射为红
    mapped[above_high] = 1.0
    # 这里clip成 0.05~0.95，使颜色更柔和
    mapped = np.clip(mapped, 0.05, 0.95)

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    H, W = mapped.shape
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(mapped, cmap=cmap, norm=norm)

    filename = f'{img_name}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=dpi)
    plt.close(fig)


def visualize_feature_mapaf(img_batch, img_name, num, save_path='feature_maps/', dpi=100,
                                     low_threshold=0.3, high_threshold=0.7, cmap_name='jet'):
    """
    支持下阈值映射蓝色、上阈值映射红色，中间正常渐变。

    参数:
    - low_threshold: float, 小于该值强制变蓝（数值映射为0）
    - high_threshold: float, 大于该值强制变红（数值映射为1）
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    feature_map = torch.squeeze(img_batch, 0).cpu()
    if len(feature_map.size()) == 2:
        feature_map = feature_map.unsqueeze(0)

    feature_map_sum = torch.sum(feature_map, dim=0)

    # 归一化
    feature_map_sum = (feature_map_sum - feature_map_sum.min()) / (feature_map_sum.max() - feature_map_sum.min())


    feature_map_sum = feature_map_sum.detach().numpy()

    # 三段映射：
    # < low_threshold => 0 (蓝)
    # > high_threshold => 1 (红)
    # 中间线性映射到0~1区间
    below_low = feature_map_sum < low_threshold
    above_high = feature_map_sum > high_threshold
    middle = (~below_low) & (~above_high)

    # 先全部置为0
    mapped = np.zeros_like(feature_map_sum)

    # 中间线性映射到0~1区间
    if high_threshold != low_threshold:
        mapped[middle] = (feature_map_sum[middle] - low_threshold) / (high_threshold - low_threshold)
    else:
        mapped[middle] = 0  # 避免除0

    # 低于下阈值的置0，映射为蓝
    mapped[below_low] = 0.0

    # 高于上阈值的置1，映射为红
    mapped[above_high] = 1.0
    # 这里clip成 0.05~0.95，使颜色更柔和
    mapped = np.clip(mapped, 0.05, 0.95)

    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    H, W = mapped.shape
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    ax.imshow(mapped, cmap=cmap, norm=norm)

    filename = f'{img_name}.png'
    plt.savefig(os.path.join(save_path, filename), dpi=dpi)
    plt.close(fig)
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
model.load_state_dict(torch.load('./trained_models_352-20/ESD/MYNet.15.pth'))

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
    save_path = './results-22222/ESD/' + dataset + '/'
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
        res, s1_sig, s2, s2_sig, s3, s3_sig, s4, s4_sig, s5, s5_sig,ff5 = model(image)
        visualize_feature_map(ff5, img_name=name, num=i, save_path='feature_maps-ESD/', dpi=100, low_threshold=0.3, high_threshold=0.7)
        # visualize_feature_map(ff5, img_name=name, num=i, save_path='feature_maps-RSDD_af/', dpi=100,
        #                       low_threshold=0.3, high_threshold=0.7)
        time_end = time.time()
        time_sum += (time_end - time_start)

        res_image = postprocess_output(res, gt.shape[-2:])
        Image.fromarray(res_image).save(save_path + name)

        if i == test_loader.size - 1:
            print('Running time {:.5f}'.format(time_sum / test_loader.size))
            print('Average speed: {:.4f} fps'.format(test_loader.size / time_sum))

