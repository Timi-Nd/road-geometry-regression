// Standalone solver checks: known answers, numerical cases and invalid inputs.
#include "../cpp/multiple_regression.h"

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using road_geometry::fit_multiple_regression;
using road_geometry::Matrix;
using road_geometry::RegressionOptions;

namespace {

void check(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected, double tolerance, const std::string& message) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

void invalid(const std::function<void()>& operation, const std::string& message) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("expected invalid_argument: " + message);
}

void exact_fit_and_prediction() {
    const Matrix x{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
                   {1, 2, 3}, {2, -1, 4}};
    std::vector<double> y;
    for (const auto& row : x) y.push_back(4 + 2 * row[0] - 3 * row[1] + 0.5 * row[2]);
    const auto model = fit_multiple_regression(x, y);
    near(model.intercept, 4, 1e-12, "intercept");
    near(model.coefficients[0], 2, 1e-12, "first coefficient");
    near(model.coefficients[1], -3, 1e-12, "second coefficient");
    near(model.coefficients[2], 0.5, 1e-12, "third coefficient");
    check(model.fitted_predictions.size() == x.size(), "fitted prediction size");
    check(model.residuals.size() == x.size(), "residual size");
    for (std::size_t i = 0; i < x.size(); ++i) {
        near(model.fitted_predictions[i], y[i], 1e-12, "fitted prediction");
        near(model.residuals[i], 0, 1e-12, "exact residual");
    }
    const auto predictions = model.predict({{4, 5, 6}, {-1, 2, 0}});
    near(predictions[0], 0, 1e-12, "new prediction");
    near(predictions[1], -4, 1e-12, "second new prediction");
    check(model.predict({}).empty(), "empty prediction set");
}

void noisy_fit() {
    const auto model = fit_multiple_regression({{-1}, {0}, {1}}, {1, 1, 4});
    near(model.intercept, 2, 1e-12, "noisy intercept");
    near(model.coefficients[0], 1.5, 1e-12, "noisy coefficient");
    near(model.residuals[0], 0.5, 1e-12, "first residual");
    near(model.residuals[1], -1, 1e-12, "second residual");
    near(model.residuals[2], 0.5, 1e-12, "third residual");
}

void origin_and_constant_feature() {
    RegressionOptions options;
    options.fit_intercept = false;
    const auto model = fit_multiple_regression({{1, 1}, {2, 1}, {3, 1}}, {5, 7, 9}, options);
    near(model.intercept, 0, 0, "fixed intercept");
    near(model.coefficients[0], 2, 1e-12, "origin coefficient");
    near(model.coefficients[1], 3, 1e-12, "constant feature without intercept");
}

void numerical_cases() {
    const Matrix scaled{{1e-9, 0}, {0, 1e9}, {1e-9, 1e9}, {2e-9, -1e9}};
    std::vector<double> y;
    for (const auto& row : scaled) y.push_back(3 + 2e9 * row[0] - 4e-9 * row[1]);
    const auto model = fit_multiple_regression(scaled, y);
    near(model.intercept, 3, 1e-12, "scaled intercept");
    near(model.coefficients[0], 2e9, 1e-5, "small-scale feature");
    near(model.coefficients[1], -4e-9, 1e-20, "large-scale feature");

    const Matrix close{{1, 1}, {2, 2 + 1e-8}, {3, 3 - 1e-8}, {4, 4 + 2e-8}};
    std::vector<double> close_y;
    for (const auto& row : close) close_y.push_back(2 * row[0] - row[1]);
    RegressionOptions options;
    options.fit_intercept = false;
    const auto close_model = fit_multiple_regression(close, close_y, options);
    near(close_model.coefficients[0], 2, 2e-7, "near-collinear coefficient 1");
    near(close_model.coefficients[1], -1, 2e-7, "near-collinear coefficient 2");
    options.rank_tolerance = 1e-6;
    invalid([&] { fit_multiple_regression(close, close_y, options); }, "configured tolerance");
}

void many_features() {
    constexpr std::size_t count = 12;
    Matrix x(count + 2, std::vector<double>(count, 0));
    std::vector<double> y(count + 2, 1);
    for (std::size_t i = 0; i < count; ++i) {
        x[i + 1][i] = 1;
        y[i + 1] += static_cast<double>(i + 1);
        x[count + 1][i] = 1;
        y[count + 1] += static_cast<double>(i + 1);
    }
    const auto model = fit_multiple_regression(x, y);
    near(model.intercept, 1, 1e-12, "many feature intercept");
    for (std::size_t i = 0; i < count; ++i) {
        near(model.coefficients[i], static_cast<double>(i + 1), 1e-12, "many feature coefficient");
    }
}

void invalid_inputs() {
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    invalid([] { fit_multiple_regression({}, {}); }, "empty");
    invalid([] { fit_multiple_regression({{}}, {1}); }, "no features");
    invalid([] { fit_multiple_regression({{1}, {2}}, {1}); }, "target length");
    invalid([] { fit_multiple_regression({{1, 2}, {3}}, {1, 2}); }, "ragged rows");
    invalid([] { fit_multiple_regression({{1, 2}}, {1}); }, "insufficient observations");
    invalid([] { fit_multiple_regression({{0}, {0}}, {1, 2}); }, "zero feature");
    invalid([] { fit_multiple_regression({{2}, {2}, {2}}, {1, 2, 3}); }, "constant feature with intercept");
    invalid([] { fit_multiple_regression({{1, 2}, {2, 4}, {3, 6}}, {1, 2, 3}); }, "dependent features");
    invalid([&] { fit_multiple_regression({{1}, {inf}}, {1, 2}); }, "infinite feature");
    invalid([&] { fit_multiple_regression({{1}, {2}}, {1, nan}); }, "nan target");
    for (double tolerance : {0.0, -1.0, 1.0, inf, nan}) {
        RegressionOptions options;
        options.rank_tolerance = tolerance;
        invalid([&] { fit_multiple_regression({{1}, {2}}, {1, 2}, options); }, "bad tolerance");
    }
    const auto model = fit_multiple_regression({{0}, {1}}, {1, 3});
    invalid([&] { model.predict({{1, 2}}); }, "prediction dimensions");
    invalid([&] { model.predict({{nan}}); }, "prediction values");
    auto invalid_model = model;
    invalid_model.coefficients[0] = nan;
    invalid([&] { invalid_model.predict({{1}}); }, "invalid coefficient");
    try {
        model.predict({{std::numeric_limits<double>::max()}});
        throw std::runtime_error("expected overflow_error");
    } catch (const std::overflow_error&) {
    }
}

}  // namespace

int main() {
    try {
        exact_fit_and_prediction();
        noisy_fit();
        origin_and_constant_feature();
        numerical_cases();
        many_features();
        invalid_inputs();
        std::cout << "All multiple regression tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Multiple regression test failed: " << error.what() << '\n';
        return 1;
    }
}
