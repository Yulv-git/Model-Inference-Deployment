<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-01-24 10:48:28
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-01-24 18:39:35
 * @FilePath: /Model_Inference_Deployment/README.md
 * @Description: Inference deployment of artificial intelligence models.
-->

<font size=5><b><big><center> Model_Inference_Deployment </center></b></big></font>

    Inference deployment of artificial intelligence models.

|                                           | Developer | Language API                                                          | Framework                                                                            | Precision Optimize | CPU/GPU/FGPA/VPU/TPU/NPU/DSP | Hardware                                                                                    | OS                                               | Application                                     | Other Features |
| ----------------------------------------- | --------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------ | ---------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------ | ----------------------------------------------- | -------------- |
| [OpenVINO](#openvino)                     | Intel     | C, C++, Python                                                        | Tensorflow, Caffe, MxNet, Keras, Pytorch, etc.                                       | INT8               | CPU, GPU, FGPA, VPU          | Intel CPU, Intel Integrated Graphics, Intel Movidius NCS, Intel Movidius VPU, DepthAI, etc. | Linux, Windows, macOS, Raspbian                  |                                                 |                |
| [TensorRT](#tensorrt)                     | NVIDIA    | C++, Python                                                           | TensorFlow, Caffe, CNTK, Chainer, Theano, Pytorch, Mxnet, PaddlePaddle, MATLAB, etc. | INT8, FP16         | GPU                          | NIVDIA GPU, NIVDIA Jetson, Tesla GPU, etc.                                                  | Linux, Windows                                   |                                                 |                |
| [MediaPipe](#mediapipe)                   | Google    | C++, JavaScript, Python                                               | TensorFlow                                                                           |                    | GPU, TPU                     | Google Coral, etc.                                                                          | Linux, Android, iOS, Raspbian                    | Youtube, Google Lens, ARCore, Google Home, etc. |                |
| [TensorFlow Lite](#tensorflow-lite)       | Google    | C++, Java, Python, Swift, Objective-C                                 | TensorFlow                                                                           | INT8, FP16         | CPU, GPU, TPU, NPU, DSP      | Google Coral, etc.                                                                          | Linux, iOS, Android, Raspberry Pi                |                                                 |                |
| [TensorFlow Serving](#tensorflow-serving) | Google    | gRPC                                                                  | TensorFlow                                                                           |                    | GPU, TPU                     |                                                                                             |                                                  |                                                 |                |
| [ONNX Runtime](#onnx-runtime)             | Microsoft | C, C++, C#, Java, JavaScript, Python, WinRT, Objective-C, Ruby, Julia | TensorFlow, Pytorch, Keras, scikit-learn, LightGBM, XGBoost                          |                    | CPU, GPU                     |                                                                                             | Linux, Windows, MacOS, iOS, Android, WebAssembly | Office 365, Bing, Visual Studio, etc.           |                |
| [Libtorch](#libtorch)                     |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [NCNN](#ncnn)                             |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [TNN](#tnn)                               |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [MNN](#mnn)                               |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [TVM](#tvm)                               |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [MACE](#mace)                             |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [Paddle Lite](#paddle-lite)               |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [MegEngine](#megengine)                   |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [OpenPPL](#openppl)                       |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [AIStation](#aistation)                   |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |
| [Bolt](#bolt)                             |           |                                                                       |                                                                                      |                    |                              |                                                                                             |                                                  |                                                 |                |

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

* Pytorch --> ONNX --> TensorRT
* Pytorch --> ONNX --> TVM
* Tensorflow --> ONNX --> NCNN
* Pytorch --> ONNX --> Tensorflow

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

Libtorch是**Facebook**开发的一套C++推理接口库，便于工业级别部署和性能优化。Pytorch可理解为是Torch的Python接口，而Libtorch则为Torch的C++接口。

## 2.8. NCNN

NCNN是**腾讯优图**于2017年7月开源的，为手机移动端极致优化的高性能神经网络前向计算框架，是业界首个移动端优化的开源神经网络推断库。
NCNN纯C++实现，无第三方依赖，可跨平台操作（覆盖了几乎所有常用的系统平台，尤其在移动平台上适用性更好，在Linux、Windows和Android、以及iOS、macOS平台上都可以使用GPU来部署模型），在手机端CPU运算速度处于领先水平。
基于NCNN平台，可轻松将深度学习算法移植到手机端并高效执行，进而产出人工智能APP，如QQ、QZone、微信、天天P图。

## 2.9. TNN

TNN是**腾讯优图**基于Rapidnet和ncnn开发的对计算、低精度、内存等进行优化的轻量级深度学习移动端推理部署工具。
TNN已在QQ、微视、P图等应用中大规模落地应用。
TNN采用统一的ONNX模型作为中转，兼容各大框架；支持FP16和int8的量化；支持计算图的优化；支持TensorFlow，PyTorch，MXNet，Caffe等多种训练框架；支持主流安卓、iOS、Embedded Linux操作系统，支持ARM CPU、GPU、NPU硬件平台；Runtime 无任何第三方库依赖，CPU 动态库尺寸仅约 400KB，并提供基础图像变换操作，调用简单便捷。

## 2.10. MNN

MNN（MobileNet Neural Network）是**阿里**轻量级深度学习移动端推理部署引擎，涵盖深度神经网络模型的优化、转换和推理。
MNN支持Tensorflow、Caffe、ONNX等主流模型格式，支持 CNN、RNN、GAN等常用网络；支持iOS 8.0+、Android 4.3+和具有POSIX接口的嵌入式设备；支持异构设备混合计算，支持CPU和GPU，可以动态导入GPU Op插件，替代CPU Op的实现；不依赖任何第三方计算库，依靠大量手写汇编实现核心运算，充分发挥ARM CPU的算力。
MNN已在淘宝、天猫、优酷、聚划算、UC、飞猪、千牛等20多个APP中使用，覆盖直播、短视频、搜索推荐、商品图像搜索、互动营销、权益发放、安全风控、天猫晚会笑脸红包、扫一扫明星猜拳大战等场景。此外，菜鸟自提柜等IOT设备中也有应用。

## 2.11. TVM

TVM是主要由**华盛顿大学**的SAMPL组贡献开发的移动端推理部署工具。

* 支持CPU、GPU和特定的加速器；
* 支持将Keras、MxNet、PyTorch、Tensorflow、CoreML、DarkNet框架的深度学习模型编译为多种硬件后端的最小可部署模型。

## 2.12. MACE

MACE（Mobile AI Compute Engine）是**小米**针对移动异构计算平台开发的深度学习推理部署框架。

## 2.13. Paddle Lite

Paddle Lite是**百度**定位于支持包括移动端、嵌入式以及服务器端在内的多硬件平台，开发的一个高性能、轻量级、灵活性强且易于扩展的深度学习推理框架。

* 支持多平台：涵盖 Android、iOS、嵌入式 Linux 设备、Windows、macOS和Linux主机；
* 支持多种语言：包括 Java、Python、C++；
* 轻量化和高性能：针对移动端设备的机器学习进行优化，压缩模型和二进制文件体积，高效推理，降低内存消耗。

## 2.14. MegEngine

MegEngine是**旷视**开发的训练推理一体化的工业级深度学习开源框架MegEngine（天元）。

* 运算速度快：MegEngine 动态、静态结合的内存优化机制，因此速度比TensorFlow更快；
* 内存占用少：通过分析整个执行方案的内存使用情况，MegEngine 充分优化内存，特别是亚线性内存优化，可以支持复杂的网络结构，自动利用部分冗余计算缩减内存占用，可达两个数量级，从而支持更大规模的模型训练；
* 易用性好：MegEngine 封装了平台细节，易于新人用户快速上手；
* 支持多种硬件平台和异构计算：MegEngine 支持通用 CPU、GPU、FPGA 以及其他移动设备端硬件，可多卡多机进行训练；
* 训练部署一体化：整个框架既可用于训练又同时支持推理，实现模型一次训练，多设备部署，避免复杂的转换过程造成的性能下降和精度损失。

## 2.15. OpenPPL

OpenPPL是**商汤**开发的用于高效人工智能推理的高性能深度学习推理引擎。它可以运行各种ONNX模型，并对OpenMMLab有更好的支持。

## 2.16. AIStation

AIStation是**浪潮**开发的人工智能推理服务平台，主要面向企业AI应用部署及在线服务管理场景，通过统一应用接口、算力弹性伸缩、A/B测试、滚动发布、多模型加权评估等全栈AI能力，为企业提供可靠、易用、灵活的推理服务部署及计算资源管理平台，帮助用户AI业务快速上线，提高AI计算资源的利用效率，实现AI产业的快速落地。

## 2.17. Bolt

Bolt是**华为**开发的用于深度学习的轻量级、高性能、异构性灵活的推理框架，适用于各种神经网络的通用部署工具。
Bolt已在华为的多个部门广泛部署和使用，如2012实验室，CBG和华为产品线。

---

<font size=4><b><big> Contributing </b></big></font>

If you find errors in this repo. or have any suggestions, please feel free to please feel free to pull requests.
