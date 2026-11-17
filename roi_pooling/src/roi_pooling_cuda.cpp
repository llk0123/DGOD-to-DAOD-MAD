#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "roi_pooling_kernel.h"

std::vector<at::Tensor> roi_pooling_forward(
    const at::Tensor& features,
    const at::Tensor& rois,
    float spatial_scale,
    int pooled_height,
    int pooled_width) {

  // 输入合法性检查
  TORCH_CHECK(features.is_cuda(), "features must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = features.size(1);
  auto height = features.size(2);
  auto width = features.size(3);

  auto output = at::zeros({num_rois, channels, pooled_height, pooled_width}, features.options());
  auto argmax = at::zeros({num_rois, channels, pooled_height, pooled_width}, at::dtype(at::kInt).device(features.device()));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ROIPoolForwardLauncher(
      features.contiguous().data_ptr<float>(),
      spatial_scale,
      num_rois,
      height,
      width,
      channels,
      pooled_height,
      pooled_width,
      rois.contiguous().data_ptr<float>(),
      output.data_ptr<float>(),
      argmax.data_ptr<int>(),
      stream);

  return {output, argmax};
}

at::Tensor roi_pooling_backward(
    const at::Tensor& top_grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
    float spatial_scale,
    int batch_size,
    int height,
    int width,
    int channels,
    int pooled_height,
    int pooled_width) {

  TORCH_CHECK(top_grad.is_cuda(), "top_grad must be a CUDA tensor");
  TORCH_CHECK(rois.is_cuda(), "rois must be a CUDA tensor");
  TORCH_CHECK(argmax.is_cuda(), "argmax must be a CUDA tensor");

  auto bottom_grad = at::zeros({batch_size, channels, height, width}, top_grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  ROIPoolBackwardLauncher(
      top_grad.contiguous().data_ptr<float>(),
      spatial_scale,
      batch_size,
      rois.size(0),
      height,
      width,
      channels,
      pooled_height,
      pooled_width,
      rois.contiguous().data_ptr<float>(),
      bottom_grad.data_ptr<float>(),
      argmax.contiguous().data_ptr<int>(),
      stream);

  return bottom_grad;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("roi_pooling_forward", &roi_pooling_forward, "ROI Pooling forward (CUDA)");
  m.def("roi_pooling_backward", &roi_pooling_backward, "ROI Pooling backward (CUDA)");
}
