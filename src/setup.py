from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="trigrad",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="trigrad._C",
            sources=[
                "trigrad/render.cu",
                "trigrad/module.cu",
                "trigrad/util.cu",
                "trigrad/tests.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
