#!/usr/bin/env python
# coding=utf-8
'''
Author: Shuangchi He / Yulv
Email: yulvchi@qq.com
Date: 2022-01-28 14:21:09
Motto: Entities should not be multiplied unnecessarily.
LastEditors: Shuangchi He
LastEditTime: 2022-04-06 11:40:23
FilePath: /Model_Inference_Deployment/src/PyTorch2ONNX/PyTorch2ONNX_Run_in_ONNX_RUNTIME.py
Description: Init from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    Export a model from PyTorch to ONNX and run it using ONNX RUNTIME.
'''
import argparse
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
import onnx
import onnxruntime

from utils import check_dir, torchtensor2numpy


class SuperResolutionNet(nn.Module):
    ''' Super Resolution model definition in PyTorch. '''
    def __init__(self, upscale_factor, inplace=False):
        super(SuperResolutionNet, self).__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))

        self.conv4 = nn.Conv2d(32, 1 * (upscale_factor ** 2), (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Increase spatial resolution with Sub-Pixel conv.
        x = self.pixel_shuffle(self.conv4(x))

        return x

    def _initialize_weights(self):
        init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
        init.orthogonal_(self.conv4.weight)


def PyTorch2ONNX(torch_model, dummy_input_to_model, onnx_save_dir, check_onnx_model=True):
    ''' Export the model. (PyTorch2ONNX) '''
    torch.onnx.export(
        torch_model,                                    # model being run.
        dummy_input_to_model,                           # model input (or a tuple for multiple inputs).
        onnx_save_dir,                                  # where to save the model (can be a file or file-like object).
        export_params=True,                             # store the trained parameter weights inside the model file.
        opset_version=10,                               # the ONNX version to export the model to.
        do_constant_folding=True,                       # whether to execute constant folding for optimization.
        input_names=['input'],                          # the model's input names.
        output_names=['output'],                        # the model's output names.
        dynamic_axes={                                  # variable length axes.
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}})

    if check_onnx_model:  # Verify the model’s structure and confirm that the model has a valid schema.
        onnx_model = onnx.load(onnx_save_dir)
        onnx.checker.check_model(onnx_model)


def Verify_ONNX_in_ONNX_RUNTIME(onnx_dir, dummy_input_to_model, torch_out):
    ''' Verify ONNX Runtime and PyTorch are computing the same value for the model. '''
    # Create an inference session.
    ort_session = onnxruntime.InferenceSession(onnx_dir)

    # Compute ONNX Runtime output prediction.
    ort_inputs = {ort_session.get_inputs()[0].name: torchtensor2numpy(dummy_input_to_model)}
    ort_outs = ort_session.run(None, ort_inputs)

    # Compare ONNX Runtime and PyTorch results.
    np.testing.assert_allclose(torchtensor2numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def Run_ONNX_in_ONNX_RUNTIME(onnx_dir, img_path, img_save_path):
    ''' Run the model on an image using ONNX Runtime. '''
    # Take the tensor representing the greyscale resized image.
    img = Image.open(img_path)
    resize = transforms.Resize([224, 224])
    img = resize(img)
    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()
    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

    # Create an inference session.
    ort_session = onnxruntime.InferenceSession(onnx_dir)

    # Run the ONNX model in ONNX Runtime.
    ort_inputs = {ort_session.get_inputs()[0].name: torchtensor2numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

    # Get the output image.
    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    # Save the image, compare this with the output image from mobile device.
    final_img.save(img_save_path)


def main(args):
    # Create the super-resolution model.
    torch_model = SuperResolutionNet(upscale_factor=3)

    # Initialize model with the pretrained weights.
    def map_location(storage, loc): return storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(
        url='https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth', map_location=map_location))

    # Set the model to inference mode.
    torch_model.eval()

    # Input to the model.
    batch_size = 1
    dummy_input_to_model = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(dummy_input_to_model)

    # Export the model. (PyTorch2ONNX)
    PyTorch2ONNX(torch_model, dummy_input_to_model, args.onnx_save_dir, args.check_onnx_model)

    # Verify ONNX Runtime and PyTorch are computing the same value for the model.
    Verify_ONNX_in_ONNX_RUNTIME(args.onnx_save_dir, dummy_input_to_model, torch_out)

    # Run the model on an image using ONNX Runtime.
    Run_ONNX_in_ONNX_RUNTIME(args.onnx_save_dir, args.img_path, args.img_save_path)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description='Export a model from PyTorch to ONNX and run it using ONNX RUNTIME.')
    parse.add_argument('--img_path', type=str, default='{}/data/cat.jpg'.format(os.path.dirname(os.path.abspath(__file__))))
    parse.add_argument('--check_onnx_model', type=bool, default=True)
    parse.add_argument('--output_dir', type=str, default='{}/output'.format(os.path.dirname(os.path.abspath(__file__))))
    args = parse.parse_args()

    check_dir(args.output_dir)
    args.onnx_save_dir = '{}/super_resolution.onnx'.format(args.output_dir)
    args.img_save_path = '{}/cat_superres_with_ort.jpg'.format(args.output_dir)

    main(args)
