"""Build the C++ regression engine and Python bindings as one extension."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup(
    name="regression_cpp",
    version="0.1.0",
    ext_modules=[
        Pybind11Extension(
            "regression_cpp",
            [
                "cpp/regression.cpp",
                "cpp/multiple_regression.cpp",
                "cpp/multiple_regression_bindings.cpp",
            ],
            depends=["cpp/multiple_regression.h", "cpp/multiple_regression_bindings.h"],
            cxx_std=17,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
