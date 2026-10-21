// Simple linear regression mathematics, exposed to Python as `regression_cpp`.
//
// The model is   y_hat = b0 + b1 * x   (one predictor x, one target y).
// b0 is the intercept (prediction when x = 0) and b1 is the slope (change in
// y_hat for a one-unit increase in x). "Least squares" means choosing b0 and
// b1 so that the sum of squared errors  sum((y - y_hat)^2)  is as small as possible.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

namespace {

// Accept any array-like from Python and convert it to contiguous doubles.
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

// A lightweight read-only window onto an array's memory (no copy).
struct View {
    const double* data;
    std::size_t size;
};

// Validate one input array: it must be 1-D and contain only finite numbers.
// NaN or infinity would silently poison every sum below, so reject them early.
View as_view(const Array& a, const char* name) {
    if (a.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be one-dimensional");
    }
    const double* p = a.data();
    const std::size_t n = static_cast<std::size_t>(a.shape(0));
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(p[i])) {
            throw std::invalid_argument(std::string(name) + " contains a non-finite value");
        }
    }
    return {p, n};
}

// Two arrays that are compared or fitted together must line up one-to-one.
void require_pair(const View& x, const View& y) {
    if (x.size != y.size) {
        throw std::invalid_argument("inputs must have the same length");
    }
    if (x.size == 0) {
        throw std::invalid_argument("inputs must not be empty");
    }
}

// Average of the values. The plain loop `sum += x` loses small digits when the
// running sum gets large. Neumaier summation keeps a correction term `c` that
// collects the digits lost in each addition and adds them back at the end.
double mean(const View& v) {
    double sum = 0.0, c = 0.0;
    for (std::size_t i = 0; i < v.size; ++i) {
        const double t = sum + v.data[i];
        // Whichever of sum / value is larger absorbs the other; recover what was lost.
        c += (std::fabs(sum) >= std::fabs(v.data[i])) ? (sum - t) + v.data[i]
                                                       : (v.data[i] - t) + sum;
        sum = t;
    }
    return (sum + c) / static_cast<double>(v.size);
}

// Ordinary least squares (line with an intercept), solved in closed form.
// Let dx = x - mean(x) and dy = y - mean(y). Then
//   Sxx = sum(dx^2)       how spread out x is
//   Sxy = sum(dx * dy)    how x and y move together
//   b1  = Sxy / Sxx       slope
//   b0  = mean(y) - b1 * mean(x)   puts the line through the point (mean x, mean y)
// Working with the centred values dx, dy (instead of raw x*y sums) avoids large
// cancelling numbers, which is more accurate in floating point.
// Returns (intercept, slope).
std::pair<double, double> fit_ols(const Array& x_in, const Array& y_in) {
    const View x = as_view(x_in, "x"), y = as_view(y_in, "y");
    require_pair(x, y);
    // One point fits infinitely many lines, so the slope would be meaningless.
    if (x.size < 2) {
        throw std::invalid_argument("at least two observations are required");
    }
    const double mx = mean(x), my = mean(y);
    double sxx = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < x.size; ++i) {
        const double dx = x.data[i] - mx;
        sxx += dx * dx;
        sxy += dx * (y.data[i] - my);
    }
    // If every x is identical the line could be any slope; dividing by zero is invalid.
    if (sxx == 0.0) {
        throw std::invalid_argument("x has zero variance; slope is undefined");
    }
    const double slope = sxy / sxx;
    return {my - slope * mx, slope};
}

// Least squares with the intercept forced to 0 (the line must pass through the
// origin, y_hat = b1 * x). Minimising sum((y - b1*x)^2) gives
//   b1 = sum(x*y) / sum(x^2)
// Here the values are NOT centred, because the line is anchored at (0, 0)
// rather than at the means.
double fit_through_origin(const Array& x_in, const Array& y_in) {
    const View x = as_view(x_in, "x"), y = as_view(y_in, "y");
    require_pair(x, y);
    double sxx = 0.0, sxy = 0.0;
    for (std::size_t i = 0; i < x.size; ++i) {
        sxx += x.data[i] * x.data[i];
        sxy += x.data[i] * y.data[i];
    }
    if (sxx == 0.0) {
        throw std::invalid_argument("x is all zeros; slope is undefined");
    }
    return sxy / sxx;
}

// Apply the fitted line to every x: y_hat = intercept + slope * x.
Array predict(const Array& x_in, double intercept, double slope) {
    const View x = as_view(x_in, "x");
    Array out(static_cast<py::ssize_t>(x.size));
    double* o = out.mutable_data();
    for (std::size_t i = 0; i < x.size; ++i) {
        o[i] = intercept + slope * x.data[i];
    }
    return out;
}

// Mean Absolute Error = average of |actual - predicted|.
// Same units as y; every error counts in proportion to its size.
double mae(const Array& actual_in, const Array& pred_in) {
    const View a = as_view(actual_in, "actual"), p = as_view(pred_in, "predicted");
    require_pair(a, p);
    double s = 0.0;
    for (std::size_t i = 0; i < a.size; ++i) s += std::fabs(a.data[i] - p.data[i]);
    return s / static_cast<double>(a.size);
}

// Root Mean Squared Error = sqrt(average of (actual - predicted)^2).
// Same units as y, but squaring makes large misses count much more than small
// ones, so RMSE is higher than MAE when a few predictions are far off.
double rmse(const Array& actual_in, const Array& pred_in) {
    const View a = as_view(actual_in, "actual"), p = as_view(pred_in, "predicted");
    require_pair(a, p);
    double s = 0.0;
    for (std::size_t i = 0; i < a.size; ++i) {
        const double e = a.data[i] - p.data[i];
        s += e * e;
    }
    return std::sqrt(s / static_cast<double>(a.size));
}

// R^2 = 1 - SSE / SST, the share of the variation in y that the model explains.
//   SSE = sum((actual - predicted)^2)    error left over after the model
//   SST = sum((actual - mean(actual))^2) error of always guessing the mean
// 1 means a perfect fit, 0 means no better than guessing the mean, and it can be
// negative on test data if the model is worse than that. Undefined (NaN) when
// all actual values are equal, because SST is then zero.
double r2(const Array& actual_in, const Array& pred_in) {
    const View a = as_view(actual_in, "actual"), p = as_view(pred_in, "predicted");
    require_pair(a, p);
    const double ma = mean(a);
    double sse = 0.0, sst = 0.0;
    for (std::size_t i = 0; i < a.size; ++i) {
        const double e = a.data[i] - p.data[i];
        const double d = a.data[i] - ma;
        sse += e * e;
        sst += d * d;
    }
    if (sst == 0.0) return std::numeric_limits<double>::quiet_NaN();
    return 1.0 - sse / sst;
}

}  // namespace

// Python bindings: each m.def exposes one C++ function under the given name.
PYBIND11_MODULE(regression_cpp, m) {
    m.doc() = "C++ implementation of simple linear regression mathematics";
    m.def("fit_ols", &fit_ols, py::arg("x"), py::arg("y"),
          "Return (intercept, slope) from ordinary least squares with an intercept.");
    m.def("fit_through_origin", &fit_through_origin, py::arg("x"), py::arg("y"),
          "Return the slope of a least-squares line constrained through the origin.");
    m.def("predict", &predict, py::arg("x"), py::arg("intercept"), py::arg("slope"),
          "Return intercept + slope * x for each x.");
    m.def("mae", &mae, py::arg("actual"), py::arg("predicted"), "Mean absolute error.");
    m.def("rmse", &rmse, py::arg("actual"), py::arg("predicted"), "Root mean squared error.");
    m.def("r2", &r2, py::arg("actual"), py::arg("predicted"), "Coefficient of determination.");
}
