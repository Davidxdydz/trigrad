from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="difftet",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="difftet._C",
            sources=[
                "difftet/util.cpp",
                "difftet/render.cu",
                "difftet/module.cpp",
                # "difftet/decompose.cu",
                "difftet/tests.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
