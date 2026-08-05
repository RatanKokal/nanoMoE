from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='custom_moe',
    install_requires=[
        "transformers>=4.51",
        "torch>=2.2",
    ],
    ext_modules=[
        CUDAExtension(
            name='custom_moe_cuda',
            # Sources live under csrc/ after the refactor.
            sources=['csrc/moe.cu', 'src/mem_manager.cpp'],
            include_dirs=['src/include'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': [
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                ],
            }
        ),
        CUDAExtension(
            name='custom_moe_legacy',
            sources=['csrc/moe_legacy.cu', 'src/mem_manager.cpp'],
            include_dirs=['src/include'],
            extra_compile_args={
                'cxx':  ['-O3'],
                'nvcc': [
                    '-O3',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                ],
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
