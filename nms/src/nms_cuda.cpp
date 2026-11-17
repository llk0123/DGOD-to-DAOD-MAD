// 替代原来的 nms_cuda.c
//#include <torch/extension.h>
//#include <vector>
//
//// CUDA 函数声明
//std::vector<at::Tensor> nms_cuda_forward(const at::Tensor boxes, float iou_threshold);
//
//// Python binding
//std::vector<at::Tensor> nms_forward(const at::Tensor boxes, float iou_threshold) {
//    return nms_cuda_forward(boxes, iou_threshold);
//}
//
//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("nms_forward", &nms_forward, "NMS forward (CUDA)");
//}


#include <torch/extension.h>
#include <vector>
#include <ATen/ATen.h>

// CUDA kernel declaration
void nms_cuda_launcher(const float* boxes, int64_t* keep, int* num_out, int boxes_num, int boxes_dim, float iou_threshold);

// Main interface
std::vector<at::Tensor> nms_forward(const at::Tensor& boxes, float iou_threshold) {
    TORCH_CHECK(boxes.device().is_cuda(), "Boxes must be a CUDA tensor");
    TORCH_CHECK(boxes.dtype() == at::kFloat, "Boxes must be float type");
    TORCH_CHECK(boxes.size(1) == 5, "Boxes must have shape [N, 5]");

    int boxes_num = boxes.size(0);
    int boxes_dim = boxes.size(1);

    // 检查boxes_dim == 5
    if (boxes_dim != 5) {
        printf("[NMS ERROR] boxes_dim must be 5, but got %d\n", boxes_dim);
        return {};
    }

    // 按 score 排序，取出得分列，然后排序 index
    auto scores = boxes.select(1, 4);
    auto sorted = std::get<1>(scores.sort(0, /*descending=*/true));
    auto boxes_sorted = boxes.index_select(0, sorted).contiguous();

    // 分配输出 keep 和 num_out
    auto keep = at::empty({boxes_num}, boxes.options().dtype(at::kLong));
    auto num_out = at::zeros({1}, boxes.options().dtype(at::kInt));

    // 调用 CUDA 核心
    nms_cuda_launcher(
        boxes_sorted.data_ptr<float>(),
        keep.data_ptr<int64_t>(),
        num_out.data_ptr<int>(),
        boxes_num,
        boxes_dim,
        iou_threshold
    );

    // 截断 keep，根据 num_out 实际数量
    auto keep_slice = keep.slice(0, 0, num_out.item<int>());

    // 恢复到原始索引（因为我们前面按 score 排过序）
    auto final_keep = sorted.index_select(0, keep_slice);

    return {final_keep, num_out};
}

// Python binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_forward", &nms_forward, "NMS forward (CUDA)");
}
