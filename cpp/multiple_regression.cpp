// Multiple linear regression by column-scaled Householder QR.
//
// For n observations and p features, the desired prediction for row i is
//   intercept + feature[i][0] * coefficient[0] + ... .
// We choose coefficients that minimize the sum of squared prediction errors.
// QR solves this least-squares problem without explicitly forming or inverting
// X-transpose times X, which can magnify numerical errors.
#include "multiple_regression.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace road_geometry {
namespace {

// Keep invalid inputs distinct from overflow during a calculation: Python
// callers receive ValueError for the former and OverflowError for the latter.
void require_finite(double value, const char* message) {
    if (!std::isfinite(value)) {
        throw std::invalid_argument(message);
    }
}

void require_finite_result(double value) {
    if (!std::isfinite(value)) {
        throw std::overflow_error("regression calculation produced a non-finite value");
    }
}

// C++ row vectors can be ragged, unlike a rectangular 2-D NumPy array.
// Validate each row before indexing it, and reject NaN or infinity.
void validate_features(const Matrix& features, std::size_t columns) {
    for (const auto& row : features) {
        if (row.size() != columns) {
            throw std::invalid_argument("feature rows must all have the expected number of columns");
        }
        for (double value : row) {
            require_finite(value, "features must contain only finite values");
        }
    }
}

}  // namespace

std::vector<double> RegressionResult::predict(const Matrix& features) const {
    validate_features(features, coefficients.size());
    require_finite(intercept, "intercept must be finite");
    for (double coefficient : coefficients) {
        require_finite(coefficient, "coefficients must be finite");
    }
    // Apply the stored intercept and feature coefficients to each new row.
    // Empty input is a valid batch and produces an empty vector.
    std::vector<double> predictions;
    predictions.reserve(features.size());
    for (const auto& row : features) {
        double prediction = intercept;
        for (std::size_t j = 0; j < coefficients.size(); ++j) {
            prediction += row[j] * coefficients[j];
        }
        require_finite_result(prediction);
        predictions.push_back(prediction);
    }
    return predictions;
}

RegressionResult fit_multiple_regression(
    const Matrix& features,
    const std::vector<double>& target,
    const RegressionOptions& options) {
    require_finite(options.rank_tolerance, "rank tolerance must be finite");
    if (options.rank_tolerance <= 0.0 || options.rank_tolerance >= 1.0) {
        throw std::invalid_argument("rank tolerance must be greater than zero and less than one");
    }
    if (features.empty()) {
        throw std::invalid_argument("at least one observation is required");
    }
    if (features.size() != target.size()) {
        throw std::invalid_argument("features and target must have the same number of observations");
    }
    const std::size_t rows = features.size();
    const std::size_t feature_count = features.front().size();
    if (feature_count == 0) {
        throw std::invalid_argument("at least one feature is required");
    }
    // An intercept is equivalent to an extra feature column filled with 1s.
    // "offset" reserves column 0 for it when requested.
    const std::size_t offset = options.fit_intercept ? 1 : 0;
    const std::size_t columns = feature_count + offset;
    if (rows < columns) {
        throw std::invalid_argument("insufficient observations for the number of fitted parameters");
    }
    validate_features(features, feature_count);
    for (double value : target) {
        require_finite(value, "target must contain only finite values");
    }

    // Build the design matrix: one row per observation, with an optional
    // leading column of 1s. Scale every column to Euclidean norm 1 so that
    // rank detection compares relative dependence rather than feature units.
    // hypot accumulates sqrt(sum(value^2)) more safely than squaring directly.
    // The saved scales are needed later to restore the original coefficients.
    Matrix qr(rows, std::vector<double>(columns));
    std::vector<double> scales(columns, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        if (offset) qr[i][0] = 1.0;
        for (std::size_t j = 0; j < feature_count; ++j) {
            qr[i][j + offset] = features[i][j];
        }
        for (std::size_t j = 0; j < columns; ++j) {
            scales[j] = std::hypot(scales[j], qr[i][j]);
            require_finite_result(scales[j]);
        }
    }
    for (std::size_t j = 0; j < columns; ++j) {
        if (scales[j] == 0.0) {
            throw std::invalid_argument("design matrix contains an all-zero feature");
        }
        for (std::size_t i = 0; i < rows; ++i) qr[i][j] /= scales[j];
    }

    // Householder QR applies a sequence of reflections to the scaled design.
    // Each reflection zeros the entries below one diagonal element; the top
    // block becomes the upper-triangular matrix R. Applying the same
    // reflections to the target preserves distances and the least-squares
    // solution. "transformed" stores that changing copy of the target.
    std::vector<double> transformed = target;
    for (std::size_t k = 0; k < columns; ++k) {
        double norm = 0.0;
        for (std::size_t i = k; i < rows; ++i) norm = std::hypot(norm, qr[i][k]);
        // The columns started with norm 1. A tiny remaining norm means this
        // column is effectively dependent on earlier columns at this tolerance.
        // This unpivoted test depends on feature order and is not an SVD test.
        if (norm <= options.rank_tolerance) {
            throw std::invalid_argument("design matrix is rank deficient at the requested tolerance");
        }

        // A unit reflector vector v describes a mirror operation H = I - 2vv^T.
        // Choose the sign to avoid subtracting nearly equal numbers when
        // constructing v; that reduces cancellation in floating-point arithmetic.
        std::vector<double> reflector(rows - k);
        for (std::size_t i = k; i < rows; ++i) reflector[i - k] = qr[i][k] / norm;
        reflector[0] += std::copysign(1.0, reflector[0]);
        double reflector_norm = 0.0;
        for (double value : reflector) reflector_norm = std::hypot(reflector_norm, value);
        for (double& value : reflector) value /= reflector_norm;

        // Apply H to each remaining matrix column. Rather than creating H,
        // compute H*x as x - 2*v*(v dot x), where x is the current column.
        for (std::size_t j = k; j < columns; ++j) {
            // Apply the identical reflection to the target vector.
            double dot = 0.0;
            for (std::size_t i = k; i < rows; ++i) dot += reflector[i - k] * qr[i][j];
            for (std::size_t i = k; i < rows; ++i) qr[i][j] -= 2.0 * reflector[i - k] * dot;
        }
        double dot = 0.0;
        for (std::size_t i = k; i < rows; ++i) dot += reflector[i - k] * transformed[i];
        require_finite_result(dot);
        for (std::size_t i = k; i < rows; ++i) {
            transformed[i] -= 2.0 * reflector[i - k] * dot;
            require_finite_result(transformed[i]);
        }
    }

    // Solve R * solution = transformed_target by back-substitution: the last
    // row has one unknown, and each preceding row uses already-solved values.
    // Then divide by the saved column norms to convert coefficients from the
    // unit-scaled design back to the original feature units.
    std::vector<double> solution(columns);
    for (std::size_t k = columns; k-- > 0;) {
        double value = transformed[k];
        for (std::size_t j = k + 1; j < columns; ++j) value -= qr[k][j] * solution[j];
        solution[k] = value / qr[k][k];
        require_finite_result(solution[k]);
    }
    for (std::size_t j = 0; j < columns; ++j) {
        solution[j] /= scales[j];
        require_finite_result(solution[j]);
    }

    // Predictions and residuals refer to the training rows, in their original
    // order. Residuals use actual minus predicted (not the opposite sign).
    RegressionResult result;
    result.intercept = offset ? solution[0] : 0.0;
    result.coefficients.assign(solution.begin() + offset, solution.end());
    result.fitted_predictions = result.predict(features);
    result.residuals.reserve(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const double residual = target[i] - result.fitted_predictions[i];
        require_finite_result(residual);
        result.residuals.push_back(residual);
    }
    return result;
}

}  // namespace road_geometry
