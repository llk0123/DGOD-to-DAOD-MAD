// 替代roi_align_cuda.c#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "roi_align_kernel.h"
#include <ATen/cuda/CUDAContext.h>

// forward declaration of CUDA launcher functions (from your .cu file)
int ROIAlignForwardLauncher(
    const float* features, float spatial_scale, int num_rois,
    int height, int width, int channels,
    int aligned_height, int aligned_width,
    const float* rois, float* output, cudaStream_t stream);

int ROIAlignBackwardLauncher(
    const float* top_grad, float spatial_scale, int batch_size, int num_rois,
    int height, int width, int channels,
    int aligned_height, int aligned_width,
    const float* rois, float* bottom_grad, cudaStream_t stream);

// wrapper function for Python → C++ forward
std::vector<at::Tensor> roi_align_forward(
    const at::Tensor& features,
    const at::Tensor& rois,
    float spatial_scale,
    int aligned_height,
    int aligned_width) {

  TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

  int num_rois = rois.size(0);
  int channels = features.size(1);
  int height = features.size(2);
  int width = features.size(3);

  auto output = at::zeros({num_rois, channels, aligned_height, aligned_width}, features.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ROIAlignForwardLauncher(
      features.contiguous().data_ptr<float>(),
      spatial_scale,
      num_rois,
      height,
      width,
      channels,
      aligned_height,
      aligned_width,
      rois.contiguous().data_ptr<float>(),
      output.data_ptr<float>(),
      stream);

  return {output};
}

// wrapper function for Python → C++ backward
at::Tensor roi_align_backward(
    const at::Tensor& top_grad,
    const at::Tensor& rois,
    float spatial_scale,
    int batch_size,
    int channels,
    int height,
    int width,
    int aligned_height,
    int aligned_width) {

  TORCH_CHECK(top_grad.is_cuda(), "top_grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

  int num_rois = rois.size(0);
  auto bottom_grad = at::zeros({batch_size, channels, height, width}, top_grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ROIAlignBackwardLauncher(
      top_grad.contiguous().data_ptr<float>(),
      spatial_scale,
      batch_size,
      num_rois,
      height,
      width,
      channels,
      aligned_height,
      aligned_width,
      rois.contiguous().data_ptr<float>(),
      bottom_grad.data_ptr<float>(),
      stream);

  return bottom_grad;
}

// PYBIND11 binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roi_align_forward, "ROIAlign forward (CUDA)");
  m.def("backward", &roi_align_backward, "ROIAlign backward (CUDA)");
}


