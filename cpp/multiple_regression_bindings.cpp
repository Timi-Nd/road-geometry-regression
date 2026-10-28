// Python/C++ bridge for the multiple-regression engine.
//
// The solver itself (multiple_regression.cpp) uses ordinary C++ containers
// and does not depend on Python. This file translates Python/NumPy inputs
// into those containers, calls the solver, and exposes the results back to
// Python through pybind11. Each observation is one row of features; the
// matching target value is at the same row index.
#include "multiple_regression_bindings.h"
#include "multiple_regression.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace py = pybind11;
namespace {

// The flags request a C-style (row-by-row), contiguous array of doubles.
// pybind11 can convert array-like inputs, including noncontiguous NumPy
// arrays, when necessary. Conversion can allocate a temporary array; this
// alias does not promise zero-copy input.
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// Convert a Python feature array of shape (observations, features) to the
// solver's Matrix (a vector of row vectors). At prediction time, a model
// needs the same number of columns it was fitted with, even for zero rows.
// During fitting the solver validates the remaining shape and value rules.
road_geometry::Matrix feature_matrix(const Array& array, std::size_t expected_columns,
                                     bool check_columns) {
    if (array.ndim() != 2) {
        throw std::invalid_argument("features must be a two-dimensional array");
    }
    const auto rows = static_cast<std::size_t>(array.shape(0));
    const auto columns = static_cast<std::size_t>(array.shape(1));
    if (check_columns && columns != expected_columns) {
        throw std::invalid_argument("features must have the same number of columns as the fitted model");
    }
    road_geometry::Matrix matrix(rows, std::vector<double>(columns));
    // Contiguous row-major storage puts element (i, j) at i * columns + j.
    // Copying makes the Matrix independent of the lifetime of the Python array.
    const double* data = array.data();
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < columns; ++j) {
            matrix[i][j] = data[i * columns + j];
        }
    }
    return matrix;
}

// Targets have one value per observation, so their Python array must be
// one-dimensional. The solver checks that its length matches the feature rows.
std::vector<double> target_vector(const Array& array) {
    if (array.ndim() != 1) {
        throw std::invalid_argument("target must be a one-dimensional array");
    }
    const auto size = static_cast<std::size_t>(array.shape(0));
    std::vector<double> target(size);
    if (size != 0) std::copy_n(array.data(), size, target.begin());
    return target;
}

Array to_array(const std::vector<double>& values) {
    // Allocate Python-owned array storage and copy the values into it.
    // In particular, changing an array returned by a model property cannot
    // change the fitted model's stored coefficients, predictions, or residuals.
    Array result(static_cast<py::ssize_t>(values.size()));
    if (!values.empty()) std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

// Called by Python's fit_multiple_regression. Python objects are inspected
// and copied while the Global Interpreter Lock (GIL) is held. The GIL is
// CPython's lock for safely accessing its interpreter and Python objects.
road_geometry::RegressionResult fit(const Array& features, const Array& target,
                                    bool fit_intercept, double rank_tolerance) {
    auto matrix = feature_matrix(features, 0, false);
    auto values = target_vector(target);
    road_geometry::RegressionOptions options;
    options.fit_intercept = fit_intercept;
    options.rank_tolerance = rank_tolerance;
    // The calculation below uses only C++ data, so other Python threads can
    // run while it executes. The scoped release reacquires the GIL when it
    // leaves scope, including if the solver throws an exception.
    py::gil_scoped_release release;
    return road_geometry::fit_multiple_regression(matrix, values, options);
}

Array predict(const road_geometry::RegressionResult& model, const Array& features) {
    auto matrix = feature_matrix(features, model.coefficients.size(), true);
    std::vector<double> predictions;
    {
        // Release the GIL only around C++ work. It is reacquired before
        // to_array creates a Python/NumPy object.
        py::gil_scoped_release release;
        predictions = model.predict(matrix);
    }
    return to_array(predictions);
}

}  // namespace

// Register the model and fitting function in the existing regression_cpp
// module. Read-only properties prevent reassignment, while returned arrays
// are separate copies rather than views into the C++ result.
void bind_multiple_regression(py::module_& module) {
    py::class_<road_geometry::RegressionResult>(module, "MultipleRegressionResult",
        "Fitted multiple regression model. Array properties are independent copies.")
        .def_property_readonly("intercept", [](const road_geometry::RegressionResult& model) {
            return model.intercept;
        })
        .def_property_readonly("coefficients", [](const road_geometry::RegressionResult& model) {
            return to_array(model.coefficients);
        })
        .def_property_readonly("fitted_predictions", [](const road_geometry::RegressionResult& model) {
            return to_array(model.fitted_predictions);
        })
        .def_property_readonly("residuals", [](const road_geometry::RegressionResult& model) {
            return to_array(model.residuals);
        })
        .def("predict", &predict, py::arg("features"),
             "Predict for a two-dimensional array of new observations.");

    // py::kw_only() makes fit_intercept and rank_tolerance keyword-only in
    // Python. pybind11 translates std::invalid_argument to ValueError and
    // std::overflow_error to OverflowError instead of returning a fake result.
    module.def("fit_multiple_regression", &fit, py::arg("features"), py::arg("target"),
               py::kw_only(), py::arg("fit_intercept") = true,
               py::arg("rank_tolerance") = 1e-12,
               "Fit multiple regression with Householder QR and return a fitted model.");
}
