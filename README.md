<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-01-24 10:48:28
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-04-06 11:42:47
 * @FilePath: /Model_Inference_Deployment/README.md
 * @Description: Inference deployment of artificial intelligence models.
 * Repository: https://github.com/Yulv-git/Model_Inference_Deployment
-->

<h1><center> Model_Inference_Deployment </center></h1>

    Inference deployment of artificial intelligence models.

| Tool | Developer | API | Framework / ONNX | Quantization | Processors / Accelerator | Hardware | OS | Application | Other Features |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [OpenVINO](https://docs.openvino.ai/latest/index.html) | Intel | C, C++, Python | TensorFlow, Caffe, MXNet, Keras, PyTorch, ONNX | INT8, FP16 | CPU, iGPU, GPU, VPU, GNA, FPGA (deprecated after 2020.4) | [Intel series devices](https://docs.openvino.ai/latest/openvino_docs_IE_DG_supported_plugins_Supported_Devices.html), Amazon Alexa Premium Far-Field Developer Kit, etc. | Linux, Windows, macOS, Raspbian  |  |  |
| [TensorRT](https://developer.nvidia.com/zh-cn/tensorrt) | NVIDIA | C++, Python | TensorFlow, Caffe, CNTK, Chainer, PyTorch, MXNet, PaddlePaddle, MATLAB, ONNX | INT8, FP16 | GPU | NIVDIA GPU, NIVDIA Jetson, Tesla GPU, etc. | Linux, Windows |  |  |
| [MediaPipe](https://google.github.io/mediapipe) | Google | C++, JavaScript, Python | TensorFlow |  | GPU, TPU | Google Coral, etc. | Linux, Android, iOS, Raspbian, macOS, Windows (experimental) | Youtube, Google Lens, ARCore, Google Home, etc. |  |
| [TensorFlow Lite](https://www.tensorflow.org/lite) | Google | C++, Java, Python, Swift, Objective-C (coming soon) | TensorFlow | INT8, FP16 | CPU, GPU, TPU, NPU, DSP | Google Coral, Microcontrollers, etc. | Linux, iOS, Android, Raspberry Pi | Google Search, Gmail, Google Translate, WPS Office, VSCO, etc. |  |
| [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) | Google | gRPC, RESTful | TensorFlow |  | GPU, TPU |  |  |  |  |
| [ONNX Runtime](https://onnxruntime.ai/about.html) | Microsoft | C, C++, C#, Java, JavaScript, Python, WinRT, Objective-C, Ruby, Julia | TensorFlow, PyTorch, Keras, SciKit Learn, LightGBM, XGBoost, ONNX | INT8, UINT8 | CPU, GPU, NPU (preview) |  | Linux, Windows, macOS, iOS, Android, WebAssembly | Office 365, Bing, Visual Studio, etc. |  |
| [LibTorch](https://pytorch.org/cppdocs/installing.html) | FaceBook | C++ | PyTorch |  | CPU, GPU |  | Linux, Windows, macOS |  |  |
| [NCNN](https://ncnn.docsforge.com) | Tencent |  | TensorFlow, Caffe, MXNet, Keras, PyTorch, ONNX | INT8, FP16 | CPU, GPU |  | Linux, Windows, Android, macOS, iOS, WebAssembly, RISC-V GCC/Newlib | QQ, QZone, WeChat, Pitu, etc. |  |
| [TNN](https://github.com/Tencent/TNN) | Tencent |  | TensorFlow, Caffe, MXNet, PyTorch, ONNX | INT8, FP16 | CPU, GPU, NPU |  | Linux, Android, iOS, Windows | mobile QQ, weishi, Pitu, etc. |  |
| [MNN](https://www.mnn.zone/index.html) | Alibaba | Python | TensorFlow, Caffe, ONNX | FP16 | CPU, GPU, NPU | embedded devices with POSIX interface, etc. | iOS, Android | Taobao, Tmall, Youku, Dingtalk, Xianyu, etc. | |
| [TVM](https://tvm.apache.org) | University of Washington | Python, Java, C++, TypeScript | TensorFlow, Keras, MXNet, PyTorch, CoreML, DarkNet, ONNX |  | CPU, GPU, NPU, DSP, FPGA | Microcontrollers, Browsers, etc. |  |  |  |
| [MACE](https://mace.readthedocs.io/en/latest/introduction.html) | Xiaomi |  | TensorFlow, Caffe, ONNX |  | CPU, GPU, DSP |  | Android, iOS, Linux, Windows |  |  |
| [Paddle Lite](https://paddle-lite.readthedocs.io/zh/develop/guide/introduction.html) | Baidu | C++, Java, Python |  PaddlePaddle | INT8, INT16 | CPU, GPU, NPU, FPGA, XPU, APU, NNA, TPU | [ARM Cortex-A family of processors, ARM Mali, Qualcomm Adreno, Apple A Series GPU](https://paddle-lite.readthedocs.io/zh/develop/quick_start/support_hardware.html#), etc. | Android, iOS, Linux, Windows, macOS |  |  |
| [MegEngine Lite](https://megengine.org.cn/doc/stable/zh/user-guide/deployment/lite/index.html) | Megvii |  Python, C, C++ | MegEngine | INT8  | CPU, GPU, FPGA, NPU |  | Linux, Windows, macOS, Android |  |  |
| [OpenPPL](https://openppl.ai/home) | SenseTime | C++, Python, Lua | ONNX | FP16 | CPU, GPU |  | Linux, RISC-V |  |  |
| [Bolt](https://huawei-noah.github.io/bolt) | Huawei | C, Java | TensorFlow, Caffe, ONNX | 1-BIT, INT8, FP16 | CPU, GPU |  | Linux, Windows, macOS, Andriod, iOS | 2012 Laboratory, CBG, HUAWEI Product Lines |  |

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
  - [2.14. MegEngine Lite](#214-megengine-lite)
  - [2.15. OpenPPL](#215-openppl)
  - [2.16. Bolt](#216-bolt)
- [3. Practice](#3-practice)
  - [3.1. ONNX](#31-onnx)
    - [3.1.1. Exporting a model from PyTorch to ONNX and running it using ONNX RUNTIME](#311-exporting-a-model-from-pytorch-to-onnx-and-running-it-using-onnx-runtime)

---

# 1. ONNX

[Official Website](https://onnx.ai) | [GitHub](https://github.com/onnx)

ONNX (Open Neural Network Exchange) is an open format built to represent machine learning models. ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

ONNX developed by Microsoft, Amazon, FaceBook, IBM, etc. [ONNX supported tools](https://onnx.ai/supported-tools.html): Caffe, CoreML, Keras, libSVM, MATLAB, MindSpore, MXNet, PaddlePaddle, PyTorch, SciKit Learn, TensorFlow, XGBoost, OpenVINO, TensorRT, ONNX MLIR, ONNX RUNTIME, MACE, NCNN, TVM, etc.

Eg:

- PyTorch --> ONNX --> ONNX RUNTIME
- PyTorch --> ONNX --> TensorRT
- PyTorch --> ONNX --> TVM
- TensorFlow --> ONNX --> NCNN
- PyTorch --> ONNX --> TensorFlow

---

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

TNN: A high-performance, lightweight neural network inference framework open sourced by Tencent Youtu Lab. It also has many outstanding advantages such as cross-platform, high performance, model compression, and code tailoring. The TNN framework further strengthens the support and performance optimization of mobile devices on the basis of the original Rapidnet and ncnn frameworks. At the same time, it refers to the high performance and good scalability characteristics of the industry's mainstream open source frameworks, and expands the support for X86 and NV GPUs. On the mobile phone, TNN has been used by many applications such as mobile QQ, weishi, and Pitu. As a basic acceleration framework for Tencent Cloud AI, TNN has provided acceleration support for the implementation of many businesses. Everyone is welcome to participate in the collaborative construction to promote the further improvement of the TNN inference framework.

## 2.10. MNN

[Official Website](https://www.mnn.zone/index.html) | [GitHub](https://github.com/alibaba/MNN)

MNN is a highly efficient and lightweight deep learning framework. It supports inference and training of deep learning models, and has industry leading performance for inference and training on-device. At present, MNN has been integrated in more than 20 apps of Alibaba Inc, such as Taobao, Tmall, Youku, Dingtalk, Xianyu and etc., covering more than 70 usage scenarios such as live broadcast, short video capture, search recommendation, product searching by image, interactive marketing, equity distribution, security risk control. In addition, MNN is also used on embedded devices, such as IoT.

## 2.11. TVM

[Official Website](https://tvm.apache.org) | [GitHub](https://github.com/apache/tvm)

Apache TVM is an open source machine learning compiler framework for CPUs, GPUs, and machine learning accelerators. It aims to enable machine learning engineers to optimize and run computations efficiently on any hardware backend.

## 2.12. MACE

[Official Website](https://mace.readthedocs.io/en/latest/introduction.html) | [GitHub](https://github.com/XiaoMi/mace)

MACE (Mobile AI Compute Engine) is a deep learning inference framework optimized for mobile heterogeneous computing platforms computing on Android, iOS, Linux and Windows devices. MACE provides tools and documents to help users to deploy deep learning models to mobile phones, tablets, personal computers and IoT devices.

## 2.13. Paddle Lite

[Official Website](https://paddle-lite.readthedocs.io/zh/develop/guide/introduction.html) | [GitHub](https://github.com/PaddlePaddle/Paddle-Lite)

Paddle Lite is an updated version of Paddle-Mobile, an open-open source deep learning framework designed to make it easy to perform inference on mobile, embeded, and IoT devices. It is compatible with PaddlePaddle and pre-trained models from other sources.

## 2.14. MegEngine Lite

[Official Website](https://megengine.org.cn/doc/stable/zh/user-guide/deployment/lite/index.html) | [GitHub](https://github.com/MegEngine/MegEngine/tree/master/lite)

MegEngine Lite is a layer of interface encapsulation for MegEngine. The main purpose of MegEngine Lite is to provide users with a more concise, easy-to-use and efficient inference interface, and to make full use of the multi-platform inference capabilities of MegEngine.

## 2.15. OpenPPL

[Official Website](https://openppl.ai/home) | [GitHub](https://github.com/openppl-public/ppl.nn)

OpenPPL is an open-source deep-learning inference platform based on self-developed high-performance kernel libraries. It enables AI applications to run efficiently on mainstream CPU and GPU platforms, delivering reliable inference services in cloud scenarios.

## 2.16. Bolt

[Official Website](https://huawei-noah.github.io/bolt) | [GitHub](https://github.com/huawei-noah/bolt)

Bolt is a light-weight library for deep learning. Bolt, as a universal deployment tool for all kinds of neural networks, aims to minimize the inference runtime as much as possible. Bolt has been widely deployed and used in many departments of HUAWEI company, such as 2012 Laboratory, CBG and HUAWEI Product Lines.

---

# 3. Practice

## 3.1. ONNX

ONNX is widely supported and can be found in many frameworks, tools, and hardware. Enabling interoperability between different frameworks and streamlining the path from research to production helps increase the speed of innovation in the AI community.

### 3.1.1. Exporting a model from PyTorch to ONNX and running it using ONNX RUNTIME

The main functions are as follows:

``` python
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
```

``` python
def Run_ONNX_in_ONNX_RUNTIME(onnx_dir, img_path, img_save_path):
    ''' Running the model on an image using ONNX Runtime. '''
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
```

And see [PyTorch2ONNX_Run_in_ONNX_RUNTIME.py](./src/PyTorch2ONNX/PyTorch2ONNX_Run_in_ONNX_RUNTIME.py) for the full Python script.

---

<font size=4><b><big> Contributing </b></big></font>

If you find any errors, or have any suggestions, please feel free to please feel free to pull requests.
