from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='add_matrix_cuda',
    version='1.0',
    author='kwea123',
    author_email='kwea123@gmail.com',
    description='cppcuda example',
    long_description='cppcuda example',
    ext_modules=[
        CUDAExtension(
            name='add_matrix_cuda',
            sources=['add_matrix.cpp', 'add_matrix_kernel.cu'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)