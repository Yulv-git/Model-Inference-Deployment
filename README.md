<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-01-24 10:48:28
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-01-24 22:30:03
 * @FilePath: /Model_Inference_Deployment/README.md
 * @Description: Inference deployment of artificial intelligence models.
 * https://github.com/Yulv-git/Model_Inference_Deployment
-->

<font size=5><b><big><center> Model_Inference_Deployment </center></b></big></font>

    Inference deployment of artificial intelligence models.

|                                           | Developer                | Language API                                                          | Framework                                                                            | Precision Optimize | CPU/GPU/FGPA/VPU/TPU/NPU/DSP/XPU/APU | Hardware                                                                                    | OS                                                                  | Application                                     | Other Features |
| ----------------------------------------- | ------------------------ | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------ | ---------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------- | -------------- |
| [OpenVINO](#openvino)                     | Intel                    | C, C++, Python                                                        | Tensorflow, Caffe, MxNet, Keras, PyTorch, etc.                                       | INT8               | CPU, GPU, FGPA, VPU          | Intel CPU, Intel Integrated Graphics, Intel Movidius NCS, Intel Movidius VPU, DepthAI, etc. | Linux, Windows, macOS, Raspbian                                     |                                                 |                |
| [TensorRT](#tensorrt)                     | NVIDIA                   | C++, Python                                                           | TensorFlow, Caffe, CNTK, Chainer, Theano, PyTorch, Mxnet, PaddlePaddle, MATLAB, etc. | INT8, FP16         | GPU                          | NIVDIA GPU, NIVDIA Jetson, Tesla GPU, etc.                                                  | Linux, Windows                                                      |                                                 |                |
| [MediaPipe](#mediapipe)                   | Google                   | C++, JavaScript, Python                                               | TensorFlow                                                                           |                    | GPU, TPU                     | Google Coral, etc.                                                                          | Linux, Android, iOS, Raspbian                                       | Youtube, Google Lens, ARCore, Google Home, etc. |                |
| [TensorFlow Lite](#tensorflow-lite)       | Google                   | C++, Java, Python, Swift, Objective-C                                 | TensorFlow                                                                           | INT8, FP16         | CPU, GPU, TPU, NPU, DSP      | Google Coral, etc.                                                                          | Linux, iOS, Android, Raspberry Pi                                   |                                                 |                |
| [TensorFlow Serving](#tensorflow-serving) | Google                   | gRPC                                                                  | TensorFlow                                                                           |                    | GPU, TPU                     |                                                                                             |                                                                     |                                                 |                |
| [ONNX Runtime](#onnx-runtime)             | Microsoft                | C, C++, C#, Java, JavaScript, Python, WinRT, Objective-C, Ruby, Julia | TensorFlow, PyTorch, Keras, scikit-learn, LightGBM, XGBoost                          |                    | CPU, GPU                     |                                                                                             | Linux, Windows, macOS, iOS, Android, WebAssembly                    | Office 365, Bing, Visual Studio, etc.           |                |
| [Libtorch](#libtorch)                     | FaceBook                 | C++                                                                   | PyTorch                                                                              |                    | CPU, GPU                     |                                                                                             | Linux, Windows, macOS                                               |                                                 |                |
| [NCNN](#ncnn)                             | Tencent                  |                                                           | TensorFlow, Caffe, MxNet, Keras, PyTorch                                             |      INT8, FP16                | CPU, GPU                     |                                                                                             | Linux, Windows, Android, macOS, iOS, WebAssembly, RISC-V GCC/Newlib | QQ, QZone, WeChat, Pitu, etc.                   |                |
| [TNN](#tnn)                               | Tencent                  |                                                            | TensorFlow, Caffe, MxNet, PyTorch                                                    | INT8, FP16                    | CPU, GPU, NPU                |                                                                                             | Linux, Android, iOS                                                 | QQ, weishi, Pitu, etc.                          |                |
| [MNN](#mnn)                               | Alibaba                  |                                                                       | TensorFlow, Caffe                                                                    |                    | CPU, GPU, NPU                |                                                                                             |                                                                     | Taobao, Tmall, Youku, Dingtalk, Xianyu, etc.    |                |
| [TVM](#tvm)                               | University of Washington |               |         TensorFlow, Keras, MxNet, PyTorch                        |             | CPU, GPU                  |                          |                      |                      |                |
| [MACE](#mace)                             |   Xiaomi                       |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                              Android, iOS, Linux, Windows       |                                                 |                |
| [Paddle Lite](#paddle-lite)               |     Baidu                     |    C++, Java, Python                                                                   |              PaddlePaddle                                                                        |                    |                  CPU, GPU, NPU, FPGA, XPU, APU            |                                                                                             |                                            Android, iOS, Linux, Windows, macOS                                 |                |
| [MegEngine](#megengine)                   |       Megvii                   |     Python                                                                  |                                                                                      |                    |              CPU, GPU, FPGA                |                                                                                             |                                                                     |                                                 |                |
| [OpenPPL](#openppl)                       |   SenseTime                       |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                                     |                                                 |                |
| [AIStation](#aistation)                   |   Inspur                       |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                                     |                                                 |                |
| [Bolt](#bolt)                             | Huawei                         |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                                     |                                                 |                |

---

<font size=4><b><center> Table of Contents </center></b></font>

- [1. ONNX](#1-onnx)
- [2. Platform](#2-platform)
  - [2.1. OpenVINO](#21-openvino)
  - [2.2. TensorRT](#22-tensorrt)
  - [2.3. MediaPipe](#23-mediapipe)
  - [2.4. TensorFlow Lite](#24-tensorflow-lite)
  - [2.5. TensorFlow Serving](#25-tensorflow-serving)
  - [2.6. ONNX Runtime](#26-onnx-runtime)
  - [2.7. Libtorch](#27-libtorch)
  - [2.8. NCNN](#28-ncnn)
  - [2.9. TNN](#29-tnn)
  - [2.10. MNN](#210-mnn)
  - [2.11. TVM](#211-tvm)
  - [2.12. MACE](#212-mace)
  - [2.13. Paddle Lite](#213-paddle-lite)
  - [2.14. MegEngine](#214-megengine)
  - [2.15. OpenPPL](#215-openppl)
  - [2.16. AIStation](#216-aistation)
  - [2.17. Bolt](#217-bolt)

---

# 1. ONNX

[ONNX](https://onnx.ai) (Open Neural Network Exchange) is an open format built to represent machine learning models.
ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

[ONNX: (Open standard for machine learning interoperability)](https://github.com/onnx/onnx) developed by Microsoft, Amazon, FaceBook, IBM, etc.

eg:

* PyTorch --> ONNX --> TensorRT
* PyTorch --> ONNX --> TVM
* Tensorflow --> ONNX --> NCNN
* PyTorch --> ONNX --> Tensorflow

# 2. Platform

## 2.1. OpenVINO

[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html) (Open Visual Inference & Neural Network Optimization) is an open-source [toolkit](https://github.com/openvinotoolkit/openvino) for optimizing and deploying AI inference.

[OpenVINO Toolkit - Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo): Pre-trained Deep Learning models and demos (high quality and extremely fast).

## 2.2. TensorRT

[NVIDIA TensorRT](https://developer.nvidia.com/zh-cn/tensorrt) is an SDK that facilitates high performance machine learning inference.

The [NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html) demonstrates how to use the C++ and Python APIs for implementing the most common deep learning layers. It shows how you can take an existing model built with a deep learning framework and build a TensorRT engine using the provided parsers.

The [TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT) contains the Open Source Software components of NVIDIA TensorRT.

## 2.3. MediaPipe

[MediaPipe](https://google.github.io/mediapipe) offers [cross-platform, customizable ML solutions for live and streaming media](https://github.com/google/mediapipe).

## 2.4. TensorFlow Lite

[TensorFlow Lite](https://tensorflow.google.cn/lite) is an open source deep learning framework for on-device inference.

[TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro) is a port of TensorFlow Lite designed to run machine learning models on DSPs, microcontrollers and other devices with limited memory.

[An awesome list of TensorFlow Lite models, samples, tutorials, tools and learning resources](https://github.com/margaretmz/awesome-tensorflow-lite)

## 2.5. TensorFlow Serving

[TensorFlow Serving](https://github.com/tensorflow/serving): A flexible, high-performance serving system for machine learning models.

## 2.6. ONNX Runtime

[ONNX Runtime](https://onnxruntime.ai/about.html) is an open source project that is designed to accelerate machine learning across a wide range of frameworks, operating systems, and hardware platforms. It enables acceleration of machine learning inferencing across all of your deployment targets using a single set of API.

[ONNX Runtime](https://github.com/microsoft/onnxruntime): cross-platform, high performance ML inferencing and training accelerator.

## 2.7. Libtorch

Libtorch (PyTorch C++ API)

[Libtorch教程](https://allentdan.github.io/2021/01/16/libtorch%E6%95%99%E7%A8%8B%EF%BC%88%E4%B8%80%EF%BC%89)

## 2.8. NCNN

[NCNN](https://github.com/Tencent/ncnn) is a high-performance neural network inference framework optimized for the mobile platform.

## 2.9. TNN

[TNN](https://github.com/Tencent/TNN) developed by Tencent Youtu Lab and Guangying Lab, a uniform deep learning inference framework for mobile、desktop and server. TNN is distinguished by several outstanding features, including its cross-platform capability, high performance, model compression and code pruning. The TNN framework further strengthens the support and performance optimization of mobile devices on the basis of the original Rapidnet and ncnn frameworks.

## 2.10. MNN

[MNN](https://www.mnn.zone/index.html)轻量级高性能推理引擎

[MNN](https://github.com/alibaba/MNN) is a highly efficient and lightweight deep learning framework. It supports inference and training of deep learning models, and has industry leading performance for inference and training on-device. At present, MNN has been integrated in more than 20 apps of Alibaba Inc, such as Taobao, Tmall, Youku, Dingtalk, Xianyu and etc., covering more than 70 usage scenarios such as live broadcast, short video capture, search recommendation, product searching by image, interactive marketing, equity distribution, security risk control. In addition, MNN is also used on embedded devices, such as IoT.

## 2.11. TVM

[TVM](https://tvm.apache.org) is an open source machine learning compiler framework for CPUs, GPUs, and machine learning accelerators. It aims to enable machine learning engineers to optimize and run computations efficiently on any hardware backend.

[TVM](https://github.com/apache/tvm) is a compiler stack for deep learning systems. It is designed to close the gap between the productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends. TVM works with deep learning frameworks to provide end to end compilation to different backends.

## 2.12. MACE

[MACE](https://github.com/XiaoMi/mace) (Mobile AI Compute Engine) is a deep learning inference framework optimized for mobile heterogeneous computing on Android, iOS, Linux and Windows devices.

## 2.13. Paddle Lite

[Paddle Lite](https://github.com/PaddlePaddle/Paddle-Lite) is an updated version of Paddle-Mobile, an open-open source deep learning framework designed to make it easy to perform inference on mobile, embeded, and IoT devices. It is compatible with PaddlePaddle and pre-trained models from other sources.

## 2.14. MegEngine

[MegEngine 1.7 文档](https://megengine.org.cn/doc/stable/zh/index.html)

[MegEngine](https://github.com/MegEngine/MegEngine) is a fast, scalable and easy-to-use deep learning framework, with auto-differentiation.

## 2.15. OpenPPL

[OpenPPL](https://openppl.ai/home) is an open-source inference engine based on self-developed neural network kernel libraries. OpenPPL supports efficient and high-performance AI inferencing on a variety of hardware platforms in cloud-native environments, and provides built-in support for OpenMMLab models.

[OpenPPL](https://github.com/openppl-public/ppl.nn) is a high-performance deep-learning inference engine for efficient AI inferencing. It can run various ONNX models and has better support for OpenMMLab.

## 2.16. AIStation

[AIStation](https://www.inspur.com/lcjtww/2526894/2526897/2563322/2563340/index.html)人工智能推理服务平台，主要面向企业AI应用部署及在线服务管理场景，通过统一应用接口、算力弹性伸缩、A/B测试、滚动发布、多模型加权评估等全栈AI能力，为企业提供可靠、易用、灵活的推理服务部署及计算资源管理平台，帮助用户AI业务快速上线，提高AI计算资源的利用效率，实现AI产业的快速落地。

## 2.17. Bolt

[Bolt](https://huawei-noah.github.io/bolt) is a light-weight library for deep learning. Bolt, as a universal deployment tool for all kinds of neural networks, aims to minimize the inference runtime as much as possible. Bolt has been widely deployed and used in many departments of HUAWEI company, such as 2012 Laboratory, CBG and HUAWEI Product Lines.

[Bolt](https://github.com/huawei-noah/bolt) is a deep learning library with high performance and heterogeneous flexibility.

---

<font size=4><b><big> Contributing </b></big></font>

If you find errors in this repo. or have any suggestions, please feel free to please feel free to pull requests.
