#ifndef _ROI_ALIGN_KERNEL
#define _ROI_ALIGN_KERNEL

#ifdef __cplusplus  // 刚加的
extern "C" {
#endif

__global__ void ROIAlignForward(const int nthreads, const float* bottom_data,
    const float spatial_scale, const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width,
    const float* bottom_rois, float* top_data);

int ROIAlignForwardLauncher(
    const float* bottom_data, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int aligned_height,
    const int aligned_width, const float* bottom_rois,
    float* top_data, cudaStream_t stream);

__global__ void ROIAlignBackward(const int nthreads, const float* top_diff,
    const float spatial_scale, const int height, const int width,
    const int channels, const int aligned_height, const int aligned_width,
    float* bottom_diff, const float* bottom_rois);

int ROIAlignBackwardLauncher(const float* top_diff, const float spatial_scale, const int batch_size, const int num_rois,
    const int height, const int width, const int channels, const int aligned_height,
    const int aligned_width, const float* bottom_rois,
    float* bottom_diff, cudaStream_t stream);

#ifdef __cplusplus // 刚加的，加上才能编译成功，没有这个会失败
}
#endif



#endif