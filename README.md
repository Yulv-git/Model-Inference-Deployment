<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-01-24 10:48:28
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-01-25 12:42:18
 * @FilePath: /Model_Inference_Deployment/README.md
 * @Description: Inference deployment of artificial intelligence models.
 * https://github.com/Yulv-git/Model_Inference_Deployment
-->

<font size=5><b><big><center> Model_Inference_Deployment </center></b></big></font>

    Inference deployment of artificial intelligence models.

| Tool | Developer | API | Framework / ONNX | Quantization | Processors / Accelerator | Hardware | OS | Application | Other Features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [OpenVINO](https://docs.openvino.ai/latest/index.html) | Intel | C, C++, Python | TensorFlow, Caffe, MXNet, Keras, PyTorch, ONNX | INT8, FP16 | CPU, iGPU, GPU, VPU, GNA, FPGA (deprecated after 2020.4) | [Intel series devices](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html), Amazon Alexa Premium Far-Field Developer Kit, etc. | Linux, Windows, macOS, Raspbian  |  |  |
| [TensorRT](https://developer.nvidia.com/zh-cn/tensorrt) | NVIDIA | C++, Python | TensorFlow, Caffe, CNTK, Chainer, PyTorch, MXNet, PaddlePaddle, MATLAB, ONNX | INT8, FP16 | GPU | NIVDIA GPU, NIVDIA Jetson, Tesla GPU, etc. | Linux, Windows |  |  |
| [MediaPipe](https://google.github.io/mediapipe) | Google | C++, JavaScript, Python | TensorFlow |  | GPU, TPU | Google Coral, etc. | Linux, Android, iOS, Raspbian, macOS, Windows (experimental) | Youtube, Google Lens, ARCore, Google Home, etc. |  |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | Google | C++, Java, Python, Swift, Objective-C (coming soon) | TensorFlow | INT8, FP16 | CPU, GPU, TPU, NPU, DSP | Google Coral, Microcontrollers, etc. | Linux, iOS, Android, Raspberry Pi | Google Search, Gmail, Google Translate, WPS Office, VSCO, etc. |  |
| [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) | Google | gRPC, RESTful | TensorFlow |  | GPU, TPU |  |  |  |  |
| [ONNX Runtime](https://onnxruntime.ai/about.html) | Microsoft | C, C++, C#, Java, JavaScript, Python, WinRT, Objective-C, Ruby, Julia | TensorFlow, PyTorch, Keras, SciKit Learn, LightGBM, XGBoost | INT8, FP16 | CPU, GPU, NPU (preview) |  | Linux, Windows, macOS, iOS, Android, WebAssembly | Office 365, Bing, Visual Studio, etc. |  |
| [LibTorch](https://pytorch.org/cppdocs/installing.html) | FaceBook | C++ | PyTorch |  | CPU, GPU |  | Linux, Windows, macOS |  |  |
| [NCNN](https://ncnn.docsforge.com) | Tencent |  | TensorFlow, Caffe, MXNet, Keras, PyTorch | INT8, FP16 | CPU, GPU |  | Linux, Windows, Android, macOS, iOS, WebAssembly, RISC-V GCC/Newlib | QQ, QZone, WeChat, Pitu, etc. |  |
| [TNN](https://github.com/Tencent/TNN) | Tencent |  | TensorFlow, Caffe, MXNet, PyTorch | INT8, FP16 | CPU, GPU, NPU |  | Linux, Android, iOS, Windows | mobile QQ, weishi, Pitu, etc. |  |
| [MNN]() | Alibaba |  | TensorFlow, Caffe |  | CPU, GPU, NPU |  | Taobao, Tmall, Youku, Dingtalk, Xianyu, etc. |  |
| [TVM](#tvm) | University of Washington |  | TensorFlow, Keras, MXNet, PyTorch |  | CPU, GPU |  |  |  |  |
| [MACE](#mace) | Xiaomi |  |  |  |  |  | Android, iOS, Linux, Windows |  |  |
| [Paddle Lite](#paddle-lite) | Baidu | C++, Java, Python |  PaddlePaddle |  | CPU, GPU, NPU, FPGA, XPU, APU |  | Android, iOS, Linux, Windows, macOS |  |
| [MegEngine](#megengine) | Megvii |  Python |  |  | CPU, GPU, FPGA |  |  |  |  |
| [OpenPPL](#openppl) | SenseTime |  |  |  |  |  |  |  |  |
| [AIStation](#aistation) | Inspur |  |  |  |  |  |  |  |  |
| [Bolt](#bolt) | Huawei |  |  |  |  |  |  |  |  |

---

<font size=4><b><center> Table of Contents </center></b></font>

- [1. ONNX](#1-onnx)
- [2. Tool](#2-tool)
  - [2.1. OpenVINO](#21-openvino)
  - [2.2. TensorRT](#22-tensorrt)
  - [2.3. MediaPipe](#23-mediapipe)
  - [2.4. TensorFlow Lite](#24-tensorflow-lite)
  - [2.5. TensorFlow Serving](#25-tensorflow-serving)
  - [2.6. ONNX Runtime](#26-onnx-runtime)
  - [2.7. LibTorch](#27-libtorch)
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

[Official Website](https://onnx.ai) | [GitHub](https://github.com/onnx)

ONNX (Open Neural Network Exchange) is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

ONNX developed by Microsoft, Amazon, FaceBook, IBM, etc. [ONNX supported tools](https://onnx.ai/supported-tools.html): Caffe, MATLAB, MXNet, PaddlePaddle, PyTorch, SciKit Learn, TensorFlow, XGBoost, OpenVINO, ONNX RUNTIME, MACE, TVM, ONNX MLIR, TensorRT, NCNN, etc.

Eg:

* PyTorch --> ONNX --> TensorRT
* PyTorch --> ONNX --> TVM
* TensorFlow --> ONNX --> NCNN
* PyTorch --> ONNX --> TensorFlow

# 2. Tool

## 2.1. OpenVINO

[Official Website](https://docs.openvino.ai/latest/index.html) | [GitHub](https://github.com/openvinotoolkit/openvino)

OpenVINO (Open Visual Inference & Neural Network Optimization) is an open-source toolkit for optimizing and deploying AI inference. It reduce resource demands and efficiently deploy on a range of Intel platforms from edge to cloud.

[Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo): Pre-trained Deep Learning models and demos (high quality and extremely fast).

## 2.2. TensorRT

[Official Website](https://developer.nvidia.com/zh-cn/tensorrt) | [GitHub](https://github.com/NVIDIA/TensorRT)

NVIDIA TensorRT is an SDK for high-performance deep learning inference. This SDK contains a deep learning inference optimizer and runtime environment that provides low latency and high throughput for deep learning inference applications.

## 2.3. MediaPipe

[Official Website](https://google.github.io/mediapipe) | [GitHub](https://github.com/google/mediapipe)

MediaPipe offers cross-platform, customizable ML solutions for live and streaming media.

## 2.4. TensorFlow Lite

[Official Website](https://www.tensorflow.org/lite) | [GitHub](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite)

TensorFlow Lite is TensorFlow's lightweight solution for mobile and embedded devices. It enables low-latency inference of on-device machine learning models with a small binary size and fast performance supporting hardware acceleration.

[TensorFlow Lite for Microcontrollers](https://github.com/tensorflow/tflite-micro): A port of TensorFlow Lite designed to run machine learning models on DSPs, microcontrollers and other devices with limited memory.

[Awesome TensorFlow Lite](https://github.com/margaretmz/awesome-tensorflow-lite): An awesome list of TensorFlow Lite models with sample apps, helpful tools and learning resources.

## 2.5. TensorFlow Serving

[Official Website](https://www.tensorflow.org/tfx/guide/serving) | [GitHub](https://github.com/tensorflow/serving)

TensorFlow Serving is a flexible, high-performance serving system for machine learning models, designed for production environments. TensorFlow Serving makes it easy to deploy new algorithms and experiments, while keeping the same server architecture and APIs. TensorFlow Serving provides out-of-the-box integration with TensorFlow models, but can be easily extended to serve other types of models and data.

## 2.6. ONNX Runtime

[Official Website](https://onnxruntime.ai/about.html) | [GitHub](https://github.com/microsoft/onnxruntime)

ONNX Runtime is an open source project that is designed to accelerate machine learning across a wide range of frameworks, operating systems, and hardware platforms. It enables acceleration of machine learning inferencing across all of your deployment targets using a single set of API. ONNX Runtime automatically parses through your model to identify optimization opportunities and provides access to the best hardware acceleration available.

ONNX Runtime also offers training acceleration, which incorporates innovations from Microsoft Research and is proven across production workloads like Office 365, Bing and Visual Studio.

## 2.7. LibTorch

[Official Website](https://pytorch.org/cppdocs/installing.html) | [LibTorch Tutorials](https://github.com/AllentDan/LibtorchTutorials)

LibTorch: C++ distributions of PyTorch.

## 2.8. NCNN

[Official Website](https://ncnn.docsforge.com) | [GitHub](https://github.com/Tencent/ncnn)

NCNN is a high-performance neural network inference computing framework optimized for mobile platforms. NCNN is deeply considerate about deployment and uses on mobile phones from the beginning of design. NCNN does not have third party dependencies. it is cross-platform, and runs faster than all known open source frameworks on mobile phone cpu. Developers can easily deploy deep learning algorithm models to the mobile platform by using efficient NCNN implementation, create intelligent APPs, and bring the artificial intelligence to your fingertips. NCNN is currently being used in many Tencent applications, such as QQ, Qzone, WeChat, Pitu and so on.

## 2.9. TNN

[GitHub](https://github.com/Tencent/TNN)

TNN: A high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab. It also has many outstanding advantages such as cross-platform, high performance, model compression, and code tailoring. The TNN framework further strengthens the support and performance optimization of mobile devices on the basis of the original Rapidnet and ncnn frameworks. At the same time, it refers to the high performance and good scalability characteristics of the industry's mainstream open source frameworks, and expands the support for X86 and NV GPUs. On the mobile phone, TNN has been used by many applications such as mobile QQ, weishi, and Pitu. As a basic acceleration framework for Tencent Cloud AI, TNN has provided acceleration support for the implementation of many businesses. Everyone is welcome to participate in the collaborative construction to promote the further improvement of the TNN reasoning framework.

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
