from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_pooling_cuda',
    ext_modules=[
        CUDAExtension(
            name='roi_pooling_cuda',
            sources=[
                'src/roi_pooling_cuda.cpp',
                'src/roi_pooling_kernel.cu',
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
