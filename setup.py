from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_SOURCES = ["render.cu", "module.cu", "util.cu", "tests.cu"]

setup(
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=[
        CUDAExtension(
            name="trigrad._C",
            sources=[f"src/trigrad/{source}" for source in CUDA_SOURCES],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
