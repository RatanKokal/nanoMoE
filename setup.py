from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_moe',
    ext_modules=[
        CUDAExtension(
            name='custom_moe_cuda',
            sources=['moe.cu'],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
