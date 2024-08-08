from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="ssepoptim_extensions_unique_argmin",
    ext_modules=[
        cpp_extension.CppExtension(
            "ssepoptim_extensions_unique_argmin", ["unique_argmin.cpp"]
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
