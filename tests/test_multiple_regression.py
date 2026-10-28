"""Verify Python bindings, numerical solutions and invalid-input handling."""

import unittest

import numpy as np

import regression_cpp as regression


class MultipleRegressionTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0, 0], [1, 0], [0, 1], [1, 2], [2, -1]], dtype=float)
        self.y = 4 + 2 * self.x[:, 0] - 3 * self.x[:, 1]

    def test_fit_and_predict(self):
        model = regression.fit_multiple_regression(self.x, self.y)
        self.assertAlmostEqual(model.intercept, 4)
        np.testing.assert_allclose(model.coefficients, [2, -3], atol=1e-12)
        np.testing.assert_allclose(model.fitted_predictions, self.y, atol=1e-12)
        np.testing.assert_allclose(model.residuals, 0, atol=1e-12)
        np.testing.assert_allclose(model.predict([[2, 3], [-1, 2]]), [-1, -4], atol=1e-12)
        self.assertEqual(model.predict(np.empty((0, 2))).shape, (0,))

    def test_arrays_are_independent_copies(self):
        model = regression.fit_multiple_regression(self.x, self.y)
        for attribute in ("coefficients", "fitted_predictions", "residuals"):
            first = getattr(model, attribute)
            expected = first.copy()
            first[:] = 99
            np.testing.assert_array_equal(getattr(model, attribute), expected)
        np.testing.assert_allclose(model.predict(self.x), self.y, atol=1e-12)
        with self.assertRaises(AttributeError):
            model.intercept = 10

    def test_numpy_compatible_and_noncontiguous_inputs(self):
        x = np.repeat(self.x, 2, axis=1)[:, ::2]
        self.assertFalse(x.flags.c_contiguous)
        y = np.column_stack((self.y, self.y))[:, 0]
        model = regression.fit_multiple_regression(x, y)
        np.testing.assert_allclose(model.coefficients, [2, -3], atol=1e-12)
        np.testing.assert_allclose(model.predict(x), y, atol=1e-12)
        model = regression.fit_multiple_regression(self.x.tolist(), self.y.tolist())
        np.testing.assert_allclose(model.predict(self.x), self.y, atol=1e-12)

    def test_no_intercept(self):
        x = np.array([[1, 0], [0, 1], [1, 2]])
        y = x @ np.array([3, -2])
        model = regression.fit_multiple_regression(x, y, fit_intercept=False)
        self.assertEqual(model.intercept, 0)
        np.testing.assert_allclose(model.coefficients, [3, -2], atol=1e-12)

    def test_reference_solution_and_residuals(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, 4))
        y = 2 + x @ np.array([1.5, -2, 0.3, 5]) + rng.normal(size=100)
        model = regression.fit_multiple_regression(x, y)
        expected, _, _, _ = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)
        np.testing.assert_allclose(np.r_[model.intercept, model.coefficients], expected, atol=1e-12)
        np.testing.assert_allclose(model.residuals, y - model.fitted_predictions, atol=1e-12)

    def test_householder_solutions_and_normal_residuals(self):
        rng = np.random.default_rng(2027)
        for rows, columns, with_intercept in ((40, 1, True), (60, 3, True),
                                               (50, 5, False), (6, 5, True)):
            with self.subTest(rows=rows, columns=columns, intercept=with_intercept):
                x = rng.normal(size=(rows, columns))
                x *= np.geomspace(1e-4, 1e4, columns)
                target = rng.normal(size=rows)
                design = np.column_stack((np.ones(rows), x)) if with_intercept else x
                model = regression.fit_multiple_regression(x, target, fit_intercept=with_intercept)
                reference, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
                self.assertEqual(rank, design.shape[1])
                actual = np.r_[model.intercept, model.coefficients] if with_intercept else model.coefficients
                np.testing.assert_allclose(actual, reference, rtol=2e-7, atol=1e-8)
                np.testing.assert_allclose(model.fitted_predictions, design @ reference, atol=1e-9)
                # Least-squares residuals are orthogonal to the design columns.
                normalized = design / np.linalg.norm(design, axis=0)
                np.testing.assert_allclose(normalized.T @ model.residuals, 0, atol=1e-10)

    def test_qr_rank_tolerance_boundary(self):
        # With orthogonal unit-norm columns the second residual norm is exactly
        # the sine of the angle between the two normalized columns.
        delta = 1e-5
        x = np.array([[1.0, 1.0], [0.0, delta], [0.0, 0.0]])
        y = np.array([3.0, 2 * delta, 0.0])
        options = {"fit_intercept": False}
        result = regression.fit_multiple_regression(x, y, rank_tolerance=delta * 0.9, **options)
        np.testing.assert_allclose(result.coefficients, [1, 2], atol=1e-10)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            regression.fit_multiple_regression(x, y, rank_tolerance=delta * 1.1, **options)

    def test_invalid_shapes_and_nonfinite_values(self):
        for x, y in (
            ([1, 2, 3], [1, 2, 3]),
            (np.zeros((2, 1, 1)), [1, 2]),
            (self.x, self.y[:, None]),
            (self.x, self.y[:-1]),
            (np.empty((0, 2)), []),
            (np.empty((3, 0)), [1, 2, 3]),
            ([[1], [1]], [1, 2]),
            ([[1], [np.nan]], [1, 2]),
            ([[1], [2]], [1, np.inf]),
            ([[1, 2], [2, 4], [3, 6]], [1, 2, 3]),
        ):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                regression.fit_multiple_regression(x, y)

    def test_prediction_validation(self):
        model = regression.fit_multiple_regression(self.x, self.y)
        for x in ([1, 2], [[1]], [[1, np.nan]], np.empty((0, 1))):
            with self.subTest(x=x), self.assertRaises(ValueError):
                model.predict(x)

    def test_tolerance_is_configurable(self):
        x = [[1, 1], [2, 2 + 1e-8], [3, 3 - 1e-8], [4, 4 + 2e-8]]
        y = np.asarray(x) @ [2, -1]
        regression.fit_multiple_regression(x, y, fit_intercept=False, rank_tolerance=1e-12)
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            regression.fit_multiple_regression(x, y, fit_intercept=False, rank_tolerance=1e-6)
        for value in (0, 1, -1, np.inf, np.nan):
            with self.subTest(value=value), self.assertRaises(ValueError):
                regression.fit_multiple_regression(self.x, self.y, rank_tolerance=value)

    def test_simple_regression_api_unchanged(self):
        x = np.array([0.0, 1.0, 2.0])
        y = 2 + 3 * x
        intercept, slope = regression.fit_ols(x, y)
        np.testing.assert_allclose([intercept, slope], [2, 3])
        np.testing.assert_allclose(regression.predict(x, intercept, slope), y)
        self.assertAlmostEqual(regression.fit_through_origin(x, 3 * x), 3)
        self.assertEqual(regression.mae(y, y), 0)
        self.assertEqual(regression.rmse(y, y), 0)
        self.assertEqual(regression.r2(y, y), 1)


if __name__ == "__main__":
    unittest.main()
