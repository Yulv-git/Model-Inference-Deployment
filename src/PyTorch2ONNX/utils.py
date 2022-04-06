#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-04-06 11:15:27
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-06 11:34:36
FilePath: /Model_Inference_Deployment/src/PyTorch2ONNX/utils.py
Description: Modify here please
'''
import os


def check_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            os.makedirs(path)


def torchtensor2numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    pass
