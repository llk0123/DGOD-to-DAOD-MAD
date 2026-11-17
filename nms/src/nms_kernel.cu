// 替代nms_cuda_kernel.cu
// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <torch/types.h>
// #include <vector>
//
// __device__ float devIoU(const float* const a, const float* const b) {
//     float x1 = max(a[0], b[0]);
//     float y1 = max(a[1], b[1]);
//     float x2 = min(a[2], b[2]);
//     float y2 = min(a[3], b[3]);
//
//     float w = max(0.0f, x2 - x1 + 1);
//     float h = max(0.0f, y2 - y1 + 1);
//     float inter = w * h;
//     float area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
//     float area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
//     return inter / (area_a + area_b - inter);
// }
//
// __global__ void nms_kernel(const float* boxes, int64_t* keep, int* num_out, int boxes_num, float iou_threshold) {
//     const int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid >= boxes_num) return;
//
//     const float* cur_box = boxes + tid * 4;
//     bool keep_flag = true;
//
//     for (int i = 0; i < tid; ++i) {
//         const float* prev_box = boxes + i * 4;
//         if (devIoU(cur_box, prev_box) > iou_threshold) {
//             keep_flag = false;
//             break;
//         }
//     }
//
//     if (keep_flag) {
//         int idx = atomicAdd(num_out, 1);
//         keep[idx] = tid;
//     }
// }
//
// std::vector<at::Tensor> nms_cuda_forward(const at::Tensor boxes, float iou_threshold) {
//     const auto boxes_num = boxes.size(0);
//
//     auto keep = at::zeros({boxes_num}, at::device(boxes.device()).dtype(at::kLong));
//     auto num_out = at::zeros({1}, at::device(boxes.device()).dtype(at::kInt));
//
//     const int threads = 512;
//     const int blocks = (boxes_num + threads - 1) / threads;
//
//     nms_kernel<<<blocks, threads>>>(boxes.contiguous().data_ptr<float>(),
//                                     keep.data_ptr<int64_t>(),
//                                     num_out.data_ptr<int>(),
//                                     boxes_num,
//                                     iou_threshold);
//
//     return {keep, num_out};
// }

// #include <ATen/ATen.h>
// #include <ATen/cuda/CUDAContext.h>
// #include <torch/types.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <vector>
// #include <cmath>
// #include <cstring>
//
// #define DIVUP(m,n) (((m) + (n) - 1) / (n))
// constexpr int threadsPerBlock = sizeof(unsigned long long) * 8;
//
// __device__ inline float devIoU(const float* const a, const float* const b) {
//     float left = max(a[0], b[0]);
//     float right = min(a[2], b[2]);
//     float top = max(a[1], b[1]);
//     float bottom = min(a[3], b[3]);
//
//     float width = max(right - left + 1.0f, 0.0f);
//     float height = max(bottom - top + 1.0f, 0.0f);
//     float interS = width * height;
//
//     float Sa = (a[2] - a[0] + 1.0f) * (a[3] - a[1] + 1.0f);
//     float Sb = (b[2] - b[0] + 1.0f) * (b[3] - b[1] + 1.0f);
//
//     return interS / (Sa + Sb - interS);
// }
//
// __global__ void nms_kernel(const int n_boxes, const float iou_threshold,
//                            const float* dev_boxes, unsigned long long* dev_mask) {
//     const int row_start = blockIdx.y;
//     const int col_start = blockIdx.x;
//
//     const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
//     const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);
//
//     __shared__ float block_boxes[threadsPerBlock * 5];
//     if (threadIdx.x < col_size) {
//         for (int i = 0; i < 5; ++i) {
//             block_boxes[threadIdx.x * 5 + i] =
//                 dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + i];
//         }
//     }
//     __syncthreads();
//
//     if (threadIdx.x < row_size) {
//         const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
//         const float* cur_box = dev_boxes + cur_box_idx * 5;
//
//         unsigned long long t = 0;
//         int start = (row_start == col_start) ? threadIdx.x + 1 : 0;
//
//         for (int i = start; i < col_size; i++) {
//             if (i >= 64) break;     // 防止越界位移
//             if (devIoU(cur_box, block_boxes + i * 5) > iou_threshold) {
//                 t |= 1ULL << i;
//             }
//         }
//
//         const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
// //         dev_mask[cur_box_idx * col_blocks + col_start] = t;
//         if (cur_box_idx < n_boxes) {
//             dev_mask[cur_box_idx * col_blocks + col_start] = t;
//         }
//     }
// }
//
// void nms_cuda_launcher(const float* boxes, int64_t* keep_out, int* num_out,
//                        int boxes_num, int boxes_dim, float iou_threshold) {
//     const int col_blocks = DIVUP(boxes_num, threadsPerBlock);
//
//     float* boxes_dev = nullptr;
//     unsigned long long* mask_dev = nullptr;
//
//     // 分配 GPU 内存并复制数据
//     cudaMalloc(&boxes_dev, boxes_num * boxes_dim * sizeof(float));
// //     cudaMemcpy(boxes_dev, boxes, boxes_num * boxes_dim * sizeof(float), cudaMemcpyDeviceToDevice);
//     // 判断 boxes 是否在 CUDA 上
//     cudaPointerAttributes attr;
//     cudaPointerGetAttributes(&attr, boxes);
//
//     cudaMemcpyKind kind;
//     #if CUDART_VERSION >= 10000
//         kind = (attr.type == cudaMemoryTypeDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
//     #else
//         kind = (attr.memoryType == cudaMemoryTypeDevice) ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
//     #endif
//
//     // 安全拷贝
//     cudaError_t copy_err = cudaMemcpy(
//         boxes_dev, boxes,
//         boxes_num * boxes_dim * sizeof(float),
//         kind
//     );
//     if (copy_err != cudaSuccess) {
//         printf("[NMS ERROR] cudaMemcpy failed: %s\n", cudaGetErrorString(copy_err));
//     }
//
// // 上面被注释的那张换成上面的这一段
//
//     cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long));
//
//     dim3 blocks(DIVUP(boxes_num, threadsPerBlock),
//                 DIVUP(boxes_num, threadsPerBlock));
//     dim3 threads(threadsPerBlock);
//
//     nms_kernel<<<blocks, threads>>>(boxes_num, iou_threshold, boxes_dev, mask_dev);
// //     cudaDeviceSynchronize();
//
//     // ✅ 检查 kernel launch 是否失败
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("[NMS CUDA ERROR] kernel launch failed: %s\n", cudaGetErrorString(err));
//     }
//
//     // ✅ 可选：同步以捕捉设备端错误（如访问越界）
//     cudaDeviceSynchronize();
//
//
//
//     std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
//     cudaMemcpy(mask_host.data(), mask_dev,
//                sizeof(unsigned long long) * boxes_num * col_blocks,
//                cudaMemcpyDeviceToHost);
//
//     std::vector<unsigned long long> remv(col_blocks, 0);
//     int num_to_keep = 0;
//
//     for (int i = 0; i < boxes_num; ++i) {
//         int nblock = i / threadsPerBlock;
//         int inblock = i % threadsPerBlock;
//
//         if (!(remv[nblock] & (1ULL << inblock))) {
//             keep_out[num_to_keep++] = i;
//             for (int j = nblock; j < col_blocks; ++j) {
//                 remv[j] |= mask_host[i * col_blocks + j];
//             }
//         }
//     }
//
//     cudaMemcpy(num_out, &num_to_keep, sizeof(int), cudaMemcpyHostToDevice);
//
//     cudaFree(boxes_dev);
//     cudaFree(mask_dev);
// }

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <cstring>

#define DIVUP(m,n) (((m) + (n) - 1) / (n))
constexpr int threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ inline float devIoU(const float* const a, const float* const b) {
    float left = max(a[0], b[0]);
    float right = min(a[2], b[2]);
    float top = max(a[1], b[1]);
    float bottom = min(a[3], b[3]);

    float width = max(right - left + 1.0f, 0.0f);
    float height = max(bottom - top + 1.0f, 0.0f);
    float interS = width * height;

    float Sa = (a[2] - a[0] + 1.0f) * (a[3] - a[1] + 1.0f);
    float Sb = (b[2] - b[0] + 1.0f) * (b[3] - b[1] + 1.0f);

    return interS / (Sa + Sb - interS);
}

__global__ void nms_kernel(const int n_boxes, const float iou_threshold,
                           const float* dev_boxes, unsigned long long* dev_mask) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;

    const int row_size = min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
    const int col_size = min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

    __shared__ float block_boxes[threadsPerBlock * 5];

    // 修复问题3：完全初始化共享内存
    if (threadIdx.x < col_size) {
        for (int i = 0; i < 5; ++i) {
            block_boxes[threadIdx.x * 5 + i] =
                dev_boxes[(threadsPerBlock * col_start + threadIdx.x) * 5 + i];
        }
    } else {
        for (int i = 0; i < 5; ++i) {
            block_boxes[threadIdx.x * 5 + i] = 0.0f;
        }
    }
    __syncthreads();

    if (threadIdx.x < row_size) {
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
        if (cur_box_idx >= n_boxes) return;  // ✅ 越界保护

        const float* cur_box = dev_boxes + cur_box_idx * 5;
        unsigned long long t = 0;
        int start = (row_start == col_start) ? threadIdx.x + 1 : 0;

        for (int i = start; i < col_size; i++) {
            if (devIoU(cur_box, block_boxes + i * 5) > iou_threshold) {
                t |= 1ULL << i;
            }
        }

        const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
//         dev_mask[cur_box_idx * col_blocks + col_start] = t;
        atomicOr(&dev_mask[cur_box_idx * col_blocks + col_start], t); // 原子操作保护
    }
}

void nms_cuda_launcher(const float* boxes, int64_t* keep_out, int* num_out,
                       int boxes_num, int boxes_dim, float iou_threshold) {
    const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

    // ✅ 空输入保护
    if (boxes_num == 0) {
        cudaMemset(num_out, 0, sizeof(int));
        return;
    }

    float* boxes_dev = nullptr;
    unsigned long long* mask_dev = nullptr;
    std::vector<int64_t> keep_host(boxes_num, 0); // 修复问题1：主机端缓冲区

    // ✅ CUDA 内存分配
    cudaError_t err;
    err = cudaMalloc(&boxes_dev, boxes_num * boxes_dim * sizeof(float));
    if (err != cudaSuccess) {
        printf("[NMS ERROR] cudaMalloc boxes_dev failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // 修复问题2：直接使用设备到设备拷贝
    err = cudaMemcpy(boxes_dev, boxes, boxes_num * boxes_dim * sizeof(float), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        printf("[NMS ERROR] cudaMemcpy boxes failed: %s\n", cudaGetErrorString(err));
        cudaFree(boxes_dev);
        return;
    }

    err = cudaMalloc(&mask_dev, boxes_num * col_blocks * sizeof(unsigned long long));
    if (err != cudaSuccess) {
        printf("[NMS ERROR] cudaMalloc mask_dev failed: %s\n", cudaGetErrorString(err));
        cudaFree(boxes_dev);
        return;
    }
    cudaMemset(mask_dev, 0, boxes_num * col_blocks * sizeof(unsigned long long));

    // 核函数配置
    const int max_blocks = 108 * 4; // 适配3090的108 SMs架构
    dim3 blocks(min(DIVUP(boxes_num, threadsPerBlock), max_blocks),
               min(DIVUP(boxes_num, threadsPerBlock), max_blocks));
    dim3 threads(threadsPerBlock);

    nms_kernel<<<blocks, threads>>>(boxes_num, iou_threshold, boxes_dev, mask_dev);

    // 错误检查
    if ((err = cudaGetLastError()) != cudaSuccess) {
        printf("[NMS ERROR] kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(boxes_dev);
        cudaFree(mask_dev);
        return;
    }

    // 拷贝结果到主机
    std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
    cudaMemcpy(mask_host.data(), mask_dev,
              sizeof(unsigned long long) * boxes_num * col_blocks,
              cudaMemcpyDeviceToHost);

    // NMS后处理
    std::vector<unsigned long long> remv(col_blocks, 0);
    int num_to_keep = 0;

    for (int i = 0; i < boxes_num; ++i) {
        int nblock = i / threadsPerBlock;
        int inblock = i % threadsPerBlock;

        if (!(remv[nblock] & (1ULL << inblock))) {
            keep_host[num_to_keep++] = i;
            for (int j = nblock; j < col_blocks; ++j) {
                remv[j] |= mask_host[i * col_blocks + j];
            }
        }
    }

    // 修复问题1：使用主机到设备拷贝
    cudaMemcpy(keep_out, keep_host.data(), num_to_keep * sizeof(int64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(num_out, &num_to_keep, sizeof(int), cudaMemcpyHostToDevice);

    // 释放资源
    cudaFree(boxes_dev);
    cudaFree(mask_dev);
}

