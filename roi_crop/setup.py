from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_crop_cuda',
    ext_modules=[
        CUDAExtension(
            name='roi_crop_cuda',
            sources=[
                'src/roi_crop_cuda.cpp',
                'src/roi_crop_cuda_kernel.cu',
            ])
    ],
    cmdclass={'build_ext': BuildExtension}
)