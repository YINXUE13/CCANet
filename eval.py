# -*- coding: utf-8 -*-
# @Time    : 2020/11/21
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang

import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

method = 'MyNet 13'
for _data_name in ['AUG']:
    mask_root = r'D:\object\my code2\Data\{}\test\masks'.format(_data_name)
    pred_root = r'D:\object\my code2\MYNet-master-65-EB0\MyNet-main\results\{}\test'.format(_data_name)
    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()

    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_name = os.path.splitext(mask_name)[0] + '.png'  # 假设预测图像是 png 格式
        pred_path = os.path.join(pred_root, pred_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            raise FileNotFoundError(f"Mask image not found or could not be read: {mask_path}")
        if pred is None:
            raise FileNotFoundError(f"Prediction image not found or could not be read: {pred_path}")

        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Sm": round(sm, 4),
        "wFm": round(wfm, 4),
        "MAE": round(mae, 4),
        "adpEm": round(em["adp"], 4),
        "meanEm": round(em["curve"].mean(), 4),
        "maxEm": round(em["curve"].max(), 4),
        "adpFm": round(fm["adp"], 4),
        "meanFm": round(fm["curve"].mean(), 4),
        "maxFm": round(fm["curve"].max(), 4),
    }

    print(results)
    with open("evalresults_ORSI.txt", "a") as file:
        file.write(method + ' ' + _data_name + ' ' + str(results) + '\n')
