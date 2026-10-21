from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="regression_cpp",
    version="0.1.0",
    ext_modules=[Pybind11Extension("regression_cpp", ["cpp/regression.cpp"], cxx_std=17)],
    cmdclass={"build_ext": build_ext},
)
