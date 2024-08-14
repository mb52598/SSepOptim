from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="ssepoptim_extensions_unique_argmin",
    version="0.0.1",
    ext_modules=[
        # cpp_extension.CUDAExtension(
        #     "ssepoptim_extensions_unique_argmin",
        #     ["unique_argmin.cpp", "unique_argmin_cuda_kernel.cu"]
        # )
        cpp_extension.CppExtension(
            "ssepoptim_extensions_unique_argmin",
            ["unique_argmin.cpp"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
