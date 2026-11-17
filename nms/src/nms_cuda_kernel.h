//#ifdef __cplusplus
//extern "C" {
//#endif
//
//void nms_cuda_compute(int* keep_out, int *num_out, float* boxes_host, int boxes_num,
//          int boxes_dim, float nms_overlap_thresh);
//
//#ifdef __cplusplus
//}
//#endif

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void nms_cuda_launcher(
    const float* boxes,
    int64_t* keep_out,
    int* num_out,
    int boxes_num,
    int boxes_dim,
    float iou_threshold);

#ifdef __cplusplus
}
#endif
