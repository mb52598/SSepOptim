from setuptools import setup
from torch.utils import cpp_extension

setup(
    name="test",
    ext_modules=[cpp_extension.CppExtension("test", ["test.cpp"])],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
