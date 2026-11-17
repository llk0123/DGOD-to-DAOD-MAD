from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nms_cuda',
    ext_modules=[
        CUDAExtension(
            name='nms_cuda',
            sources=['src/nms_cuda.cpp', 'src/nms_kernel.cu'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)

