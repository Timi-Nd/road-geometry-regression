#ifndef ROAD_GEOMETRY_MULTIPLE_REGRESSION_H
#define ROAD_GEOMETRY_MULTIPLE_REGRESSION_H

#include <vector>

namespace road_geometry {

// One row is an observation; columns are numerical features in a fixed order.
// For example, {{1.2, 80}, {0.5, 50}} describes two observations, each with
// two features. Every row must have the same number of features.
using Matrix = std::vector<std::vector<double>>;

struct RegressionOptions {
    // When true, fit a constant term in addition to the feature coefficients.
    // When false, the intercept is fixed at zero.
    bool fit_intercept = true;
    // Reject a scaled column if its remaining norm after preceding columns
    // have been eliminated is at or below this threshold. Must be in (0, 1).
    // The test is order-dependent and is not an SVD condition-number bound.
    double rank_tolerance = 1e-12;
};

struct RegressionResult {
    // Prediction = intercept + sum(features[j] * coefficients[j]).
    double intercept = 0.0;
    // Coefficients follow the input feature order.
    std::vector<double> coefficients;
    // These two arrays correspond to the training observations, in input order.
    std::vector<double> fitted_predictions;
    // A residual is the observed target minus its fitted prediction.
    std::vector<double> residuals;

    // Predict new rows in the same feature order; an empty batch is allowed.
    // Rows of the wrong width or containing non-finite numbers are rejected.
    std::vector<double> predict(const Matrix& features) const;
};

// Fit y = intercept + sum(feature[j] * coefficient[j]) by least squares.
// Least squares chooses coefficients that minimize the sum of squared residuals.
// Each target corresponds to a feature row; input values must be finite.
// The design (including the intercept column, if enabled) must have enough
// rows and independent columns at the requested tolerance. Invalid inputs
// throw std::invalid_argument; non-finite intermediate results throw
// std::overflow_error. See multiple_regression.cpp for the QR algorithm.
RegressionResult fit_multiple_regression(
    const Matrix& features,
    const std::vector<double>& target,
    const RegressionOptions& options = {});

}  // namespace road_geometry

#endif
