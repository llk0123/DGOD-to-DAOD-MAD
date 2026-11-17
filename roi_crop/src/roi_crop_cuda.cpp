//#include <torch/extension.h>
//#include <pybind11/pybind11.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <cuda_runtime.h>
//#include "roi_crop_cuda_kernel.h"
//
//
//std::vector<at::Tensor> roi_crop_forward(
//    at::Tensor input,
//    at::Tensor grid) {
//
//  // [N, C, H, W] — consistent with ROI output format
//  auto output = at::zeros(
//      {grid.size(0), grid.size(1), grid.size(2), input.size(1)},
//      grid.options()
//  );
//  auto stream = at::cuda::getCurrentCUDAStream();
//
//  BilinearSamplerBHWD_updateOutput_cuda_kernel(
//    output.size(1), output.size(3), output.size(2), output.size(0),
//    input.size(1), input.size(2), input.size(3), input.size(0),
//    input.data_ptr<float>(),
//    input.stride(0), input.stride(1), input.stride(2), input.stride(3),
//    grid.data_ptr<float>(),
//    grid.stride(0), grid.stride(3), grid.stride(1), grid.stride(2),
//    output.data_ptr<float>(),
//    output.stride(0), output.stride(1), output.stride(2), output.stride(3),
//    stream
//  );
//
//  return {output};
//}
//
//// backward interface
//std::vector<at::Tensor> roi_crop_backward(
//    at::Tensor input,
//    at::Tensor grid,
//    at::Tensor grad_output) {
//
//  auto grad_input = at::zeros_like(input);
//  auto grad_grid = at::zeros_like(grid);
//  auto stream = at::cuda::getCurrentCUDAStream();
//
//  BilinearSamplerBHWD_updateGradInput_cuda_kernel(
//    grad_output.size(1), grad_output.size(3), grad_output.size(2), grad_output.size(0),
//    input.size(1), input.size(2), input.size(3), input.size(0),
//    input.data_ptr<float>(),
//    input.stride(0), input.stride(1), input.stride(2), input.stride(3),
//    grid.data_ptr<float>(),
//    grid.stride(0), grid.stride(3), grid.stride(1), grid.stride(2),
//    grad_input.data_ptr<float>(),
//    grad_input.stride(0), grad_input.stride(1), grad_input.stride(2), grad_input.stride(3),
//    grad_grid.data_ptr<float>(),
//    grad_grid.stride(0), grad_grid.stride(3), grad_grid.stride(1), grad_grid.stride(2),
//    grad_output.data_ptr<float>(),
//    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
//    stream
//  );
//
//  return {grad_input, grad_grid};
//}
//
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//  m.def("forward", &roi_crop_forward, "ROI Crop forward (CUDA)");
//  m.def("backward", &roi_crop_backward, "ROI Crop backward (CUDA)");
//}         // 这个是有错误的，要使用下面这个


// gpt给的
#include <torch/extension.h>
#include <vector>
#include <ATen/cuda/CUDAContext.h>
#include "roi_crop_cuda_kernel.h"

std::vector<at::Tensor> roi_crop_forward(
    const at::Tensor& input,    // [B, H, W, C] （BHWC 格式）
    const at::Tensor& grid      // [N, H, W, 2] （BHW(YX) 坐标）
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(grid.is_cuda(), "grid must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(grid.dtype() == torch::kFloat32, "grid must be float32");
    TORCH_CHECK(input.dim() == 4, "input must be a 4D BHWC tensor");
    TORCH_CHECK(grid.dim() == 4 && grid.size(3) == 2, "grid must be of shape [N, H, W, 2]");

    // 🔧 修复：正确创建输出张量
    // 输出形状应该是 [num_rois, grid_height, grid_width, channels]
    const int num_rois = grid.size(0);
    const int grid_height = grid.size(1);
    const int grid_width = grid.size(2);
    const int channels = input.size(3);

    auto output = at::zeros({num_rois, grid_height, grid_width, channels},
                           input.options());

    const int ob = grid.size(0);    // num_rois
    const int oh = grid.size(1);    // grid_height
    const int ow = grid.size(2);    // grid_width
    const int oc = channels;        // channels
    const int ib = input.size(0);   // input_batch
    const int ih = input.size(1);   // input_height
    const int iw = input.size(2);   // input_width
    const int ic = input.size(3);   // input_channels

    // 🔧 修复：计算每张图像的ROI数量
    TORCH_CHECK(ob % ib == 0, "Number of ROIs must be divisible by batch size");
    int roiPerImage = ob / ib;

    int success = BilinearSamplerBHWD_updateOutput_cuda_kernel(
        oc, ow, oh, ob,
        ic, ih, iw, ib,
        input.data_ptr<float>(),
        input.stride(0), input.stride(3), input.stride(1), input.stride(2),
        grid.data_ptr<float>(),
        grid.stride(0), grid.stride(3), grid.stride(1), grid.stride(2),
        output.data_ptr<float>(),
        output.stride(0), output.stride(3), output.stride(1), output.stride(2),
        at::cuda::getCurrentCUDAStream()
    );

    TORCH_CHECK(success, "ROI crop forward kernel failed");
    return {output};
}

std::vector<at::Tensor> roi_crop_backward(
    const at::Tensor& input,
    const at::Tensor& grid,
    const at::Tensor& grad_output
) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(grid.is_cuda(), "grid must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(grid.dtype() == torch::kFloat32, "grid must be float32");
    TORCH_CHECK(grad_output.dtype() == torch::kFloat32, "grad_output must be float32");
    TORCH_CHECK(input.dim() == 4 && grad_output.dim() == 4, "input and grad_output must be 4D");
    TORCH_CHECK(grid.dim() == 4 && grid.size(3) == 2, "grid must be of shape [N, H, W, 2]");

    auto grad_input = at::zeros_like(input);
    auto grad_grid = at::zeros_like(grid);

    const int gob = grad_output.size(0);  // num_rois
    const int goh = grad_output.size(1);  // grid_height
    const int gow = grad_output.size(2);  // grid_width
    const int goc = grad_output.size(3);  // channels
    const int ib = input.size(0);         // input_batch
    const int ih = input.size(1);         // input_height
    const int iw = input.size(2);         // input_width
    const int ic = input.size(3);         // input_channels

    TORCH_CHECK(gob % ib == 0, "Number of ROIs must be divisible by batch size");
    int roiPerImage = gob / ib;

    int success = BilinearSamplerBHWD_updateGradInput_cuda_kernel(
        goc, gow, goh, gob,
        ic, ih, iw, ib,
        input.data_ptr<float>(),
        input.stride(0), input.stride(3), input.stride(1), input.stride(2),
        grid.data_ptr<float>(),
        grid.stride(0), grid.stride(3), grid.stride(1), grid.stride(2),
        grad_input.data_ptr<float>(),
        grad_input.stride(0), grad_input.stride(3), grad_input.stride(1), grad_input.stride(2),
        grad_grid.data_ptr<float>(),
        grad_grid.stride(0), grad_grid.stride(3), grad_grid.stride(1), grad_grid.stride(2),
        grad_output.data_ptr<float>(),
        grad_output.stride(0), grad_output.stride(3), grad_output.stride(1), grad_output.stride(2),
        at::cuda::getCurrentCUDAStream()
    );

    TORCH_CHECK(success, "ROI crop backward kernel failed");
    return {grad_input, grad_grid};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &roi_crop_forward, "ROI Crop forward (CUDA)");
    m.def("backward", &roi_crop_backward, "ROI Crop backward (CUDA)");
}


// deepseek给的
//#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>
//#include <pybind11/pybind11.h>
//#include "roi_crop_cuda_kernel.h"
//
//#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
//
//// 前向传播函数
//at::Tensor roi_crop_forward(
//    const at::Tensor& input,   // 输入特征图 [N, C, H, W]
//    const at::Tensor& grid)    // 网格坐标 [B, H, W, 2]
//{
//    CHECK_INPUT(input);
//    CHECK_INPUT(grid);
//
//    // 显式维度检查
//    TORCH_CHECK(input.size(0) == grid.size(0), "Batch size mismatch between input and grid");
//    TORCH_CHECK(input.dim() == 4, "Input must be 4D tensor");
//    TORCH_CHECK(grid.size(3) == 2, "Grid last dim must be 2");
//
//    // 计算输出尺寸 [B, grid_H, grid_W, C]
//    const int ob = grid.size(0);  // 输出batch size
//    const int oh = grid.size(1);  // 输出高度
//    const int ow = grid.size(2);  // 输出宽度
//    const int oc = input.size(1); // 输出通道数
//
//    // 显式创建NHWC格式的输出张量
//    auto output = at::empty({ob, oh, ow, oc},
//                          input.options().memory_format(c10::MemoryFormat::Contiguous));
//
//    // 获取CUDA流
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//
//    // 调用CUDA内核函数（参数顺序修正）
//    BilinearSamplerBHWD_updateOutput_cuda_kernel(
//        oc,          // output_channels
//        ow,          // output_width
//        oh,          // output_height
//        ob,          // output_batch
//        input.size(1), // input_channels
//        input.size(2), // input_height
//        input.size(3), // input_width
//        input.size(0), // input_batch
//        input.data_ptr<float>(),
//        input.stride(0),  // N stride
//        input.stride(1),  // C stride
//        input.stride(2),  // H stride
//        input.stride(3),  // W stride
//        grid.data_ptr<float>(),
//        grid.stride(0),   // N stride
//        grid.stride(3),   // YX stride (保持原始实现约定)
//        grid.stride(1),   // H stride
//        grid.stride(2),   // W stride
//        output.data_ptr<float>(),
//        output.stride(0), // N stride
//        output.stride(3), // C stride (NHWC格式)
//        output.stride(1), // H stride
//        output.stride(2), // W stride
//        stream
//    );
//
//    return output;
//}
//
//// 反向传播函数
//std::tuple<at::Tensor, at::Tensor> roi_crop_backward(
//    const at::Tensor& grad_output,
//    const at::Tensor& input,
//    const at::Tensor& grid)
//{
//    CHECK_INPUT(grad_output);
//    CHECK_INPUT(input);
//    CHECK_INPUT(grid);
//
//    auto grad_input = at::zeros_like(input);
//    auto grad_grid = at::zeros_like(grid);
//
//    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//
//    // 调用反向传播内核（参数顺序修正）
//    BilinearSamplerBHWD_updateGradInput_cuda_kernel(
//        grad_output.size(3), // grad_output_channels
//        grad_output.size(2), // grad_output_width
//        grad_output.size(1), // grad_output_height
//        grad_output.size(0), // grad_output_batch
//        input.size(1),       // input_channels
//        input.size(2),       // input_height
//        input.size(3),       // input_width
//        input.size(0),       // input_batch
//        input.data_ptr<float>(),
//        input.stride(0),     // N stride
//        input.stride(1),     // C stride
//        input.stride(2),     // H stride
//        input.stride(3),     // W stride
//        grid.data_ptr<float>(),
//        grid.stride(0),      // N stride
//        grid.stride(3),      // YX stride
//        grid.stride(1),      // H stride
//        grid.stride(2),      // W stride
//        grad_input.data_ptr<float>(),
//        grad_input.stride(0),
//        grad_input.stride(1),
//        grad_input.stride(2),
//        grad_input.stride(3),
//        grad_grid.data_ptr<float>(),
//        grad_grid.stride(0),
//        grad_grid.stride(3),
//        grad_grid.stride(1),
//        grad_grid.stride(2),
//        grad_output.data_ptr<float>(),
//        grad_output.stride(0),
//        grad_output.stride(3),
//        grad_output.stride(1),
//        grad_output.stride(2),
//        stream
//    );
//
//    return std::make_tuple(grad_input, grad_grid);
//}
//
//// PYBIND11绑定
//namespace py = pybind11;
//
//PYBIND11_MODULE(roi_crop_cuda, m) {
//    m.def("forward", &roi_crop_forward, "ROI crop forward (CUDA)");
//    m.def("backward", &roi_crop_backward, "ROI crop backward (CUDA)");
//}
