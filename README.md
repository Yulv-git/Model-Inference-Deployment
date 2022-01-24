<!--
 * @Author: Shuangchi He / Yulv
 * @Email: yulvchi@qq.com
 * @Date: 2022-01-24 10:48:28
 * @Motto: Entities should not be multiplied unnecessarily.
 * @LastEditors: Shuangchi He
 * @LastEditTime: 2022-01-24 15:06:12
 * @FilePath: /Model_Inference_Deployment/README.md
 * @Description: Modify here please
-->

# Model_Inference_Deployment

    Inference deployment of artificial intelligence models.

|                         | Developer       | Language API | Framework                                                                            | Precision Optimize | CPU/GPU/FGPA/VPU/TPU | Hardware                                                                           | OS                              | Application | Other Features |
| ----------------------- | --------------- | ------------ | ------------------------------------------------------------------------------------ | ------------------ | -------------------- | ---------------------------------------------------------------------------------- | ------------------------------- | ----------- | -------------- |
| [OpenVINO](#openvino)   | Intel           | C++, Python  | Tensorflow, Caffe, MxNet, Keras, Pytorch, etc.                                       | INT8               | CPU, GPU, FGPA, VPU  | Intel CPU, Intel Integrated Graphics, Intel Movidius NCS, Intel Movidius VPU, etc. | Linux, Windows, macOS, Raspbian |             |                |
| [TensorRT](#tensorrt)   | NVIDIA          | C++, Python  | TensorFlow, Caffe, CNTK, Chainer, Theano, Pytorch, Mxnet, PaddlePaddle, MATLAB, etc. | INT8, FP16         | GPU                  | NIVDIA GPU, NIVDIA Jetson, Tesla GPU, etc.                                         | Linux, Windows                  |             |                |
| [MediaPipe](#mediapipe) | Google Research |              | TensorFlow                                                                           |                    | TPU                  |                                                                                    |                                 |             |                |
|                         |                 |              |                                                                                      |                    |                      |                                                                                    |                                 |             |                |
|                         |                 |              |                                                                                      |                    |                      |                                                                                    |                                 |             |                |
|                         |                 |              |                                                                                      |                    |                      |                                                                                    |                                 |             |                |
|                         |                 |              |                                                                                      |                    |                      |                                                                                    |                                 |             |                |
|                         |                 |              |                                                                                      |                    |                      |                                                                                    |                                 |             |                |
|                         |                 |              |                                                                                      |                    |                      |                                                                                    |                                 |             |                |
|                         |                 |              |                                                                                      |                    |                      |                                                                                    |                                 |             |                |
|                         |                 |              |                                                                                      |                    |                      |                                                                                    |                                 |             |                |

---

<font size=4><b><center> Table of Contents </center></b></font>

- [Model_Inference_Deployment](#model_inference_deployment)
  - [ONNX](#onnx)
  - [OpenVINO](#openvino)
  - [TensorRT](#tensorrt)
  - [MediaPipe](#mediapipe)

---

## ONNX

[ONNX](https://onnx.ai) (Open Neural Network Exchange) is an open format built to represent machine learning models.
ONNX defines a common set of operators - the building blocks of machine learning and deep learning models - and a common file format to enable AI developers to use models with a variety of frameworks, tools, runtimes, and compilers.

[Open standard for machine learning interoperability](https://github.com/onnx/onnx)

eg:

* Pytorch --> ONNX --> TensorRT
* Pytorch --> ONNX --> TVM
* Tensorflow --> ONNX --> NCNN
* Pytorch --> ONNX --> Tensorflow

## OpenVINO

[OpenVINO](https://docs.openvinotoolkit.org/latest/index.html) (Open Visual Inference & Neural Network Optimization)

[OpenVINO Toolkit](https://github.com/openvinotoolkit/openvino)

[OpenVINO Toolkit - Open Model Zoo repository](https://github.com/openvinotoolkit/open_model_zoo): Pre-trained Deep Learning models and demos (high quality and extremely fast).

## TensorRT

[NVIDIA TensorRT](https://developer.nvidia.com/zh-cn/tensorrt)

[NVIDIA TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)

[TensorRT Open Source Software](https://github.com/NVIDIA/TensorRT)

## MediaPipe
