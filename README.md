# tiny-cnn

We sometimes need AI models for their outstanding performance on devices with very limited resources, while keeping them extremely lightweight. Many inference frameworks, e.g., [ORT](https://github.com/microsoft/onnxruntime) and [ncnn](https://github.com/Tencent/ncnn), perform well for larger models but fall short with tiny models: for tiny models, the initialization overhead can even far exceed the inference time, and their optimizations for large models do not always benefit smaller ones. This is where custom implementations are better.

This repo shows a tiny CNN implementation for recognizing 16x16 optical character images, which is simple, powerful enough, while extremely fast and lightweight. Although the example here is a CNN, the approach also works well for other architectures like autoencoders. As long as the model is small, the implementation is worthwhile.

The repo provides two versions: [cnn_struct.c](https://github.com/Avafly/tiny-cnn/blob/8907b70a169a4aaf4d7a55967564620e2e4d4840/cnn_struct.c) builds the model dynamically from configuration, and [cnn_const.c](https://github.com/Avafly/tiny-cnn/blob/8907b70a169a4aaf4d7a55967564620e2e4d4840/cnn_const.c) uses a fixed model architecture for the fastest inference. Compared to ONNXRuntime and ncnn, tiny CNN shows clear advantages in both speed and peak memory.

## Benchmarks

Tested on a Raspberry Pi 4 model B (4GB RAM), utilizing 4 threads for all to recognize 1,000 images.

### Speed

|     Model      | Elapsed times |
| :------------: | :-----------: |
| **cnn_const**  |  **8.91 ms**  |
| **cnn_struct** | **14.09 ms**  |
|    cnn_ort     |   32.29 ms    |
|    cnn_ncnn    |   29.61 ms    |

### Peak memory

Peak memory measured by `valgrind --tool=massif`.

|     Model      |  Peak memory  |
| :------------: | :-----------: |
| **cnn_const**  | **5.516 KiB** |
| **cnn_struct** | **5.516 KiB** |
|    cnn_ort     |   10.39 MiB   |
|    cnn_ncnn    |   417.6 KiB   |

Compared to ORT and ncnn, tiny-cnn achieves **2-3x** speedup with **negligible** peak memory usage.

## How to run

```bash
# cnn_const.c & cnn_struct.c
./run ../ModelParam.txt ../ImageData.txt

# cnn_ort.cpp
./run ../models/model.onnx ../ImageData.txt

# cnn_ncnn.cpp
./run ../models/model ../ImageData.txt
```

## References

https://github.com/BVLC/caffe

https://github.com/OpenMathLib/OpenBLAS