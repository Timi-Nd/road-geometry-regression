#ifndef ROAD_GEOMETRY_MULTIPLE_REGRESSION_BINDINGS_H
#define ROAD_GEOMETRY_MULTIPLE_REGRESSION_BINDINGS_H

#include <pybind11/pybind11.h>

// Add the multiple-regression Python class and function to an existing module.
// regression.cpp calls this while initializing the regression_cpp extension.
void bind_multiple_regression(pybind11::module_& module);

#endif
