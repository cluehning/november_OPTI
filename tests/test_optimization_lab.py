"""Numerische Tests für das interaktive Optimierungslabor."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.optimization_lab import (
    Quadratic,
    QuadraticLaboratory,
    armijo_backtracking,
    generalized_spectrum,
    metric_condition_number,
    metric_gradient,
    metric_norm,
    metric_summary,
    optimize,
    steepest_descent_direction,
    validate_metric,
)


class QuadraticTests(unittest.TestCase):
    def setUp(self):
        self.H = np.array([[8.0, 1.0], [1.0, 3.0]])
        self.b = np.array([-2.0, 1.5])
        self.objective = Quadratic(self.H, self.b, 2.25)

    def test_quadratic_gradient_and_hessian(self):
        point = np.array([0.35, -1.2])
        direction = np.array([0.7, 0.4])
        epsilon = 1e-6
        difference = (
            self.objective.value(point + epsilon * direction)
            - self.objective.value(point - epsilon * direction)
        ) / (2 * epsilon)
        self.assertAlmostEqual(
            difference, self.objective.gradient(point) @ direction, places=7
        )
        np.testing.assert_allclose(self.objective.hessian(), self.H)
        np.testing.assert_allclose(
            self.objective.gradient(self.objective.minimizer), np.zeros(2), atol=1e-12
        )

    def test_metric_norm_and_gradient(self):
        metric = np.array([[2.0, 0.4], [0.4, 1.5]])
        vector = np.array([1.5, -0.7])
        self.assertAlmostEqual(metric_norm(vector, metric) ** 2, vector @ metric @ vector)
        np.testing.assert_allclose(
            metric_gradient(vector, metric), np.linalg.solve(metric, vector)
        )

    def test_steepest_direction_is_unit_and_strictly_descending(self):
        metric = np.array([[2.0, 0.4], [0.4, 1.5]])
        gradient = np.array([1.2, -2.3])
        direction = steepest_descent_direction(gradient, metric)
        self.assertAlmostEqual(metric_norm(direction, metric), 1.0)
        self.assertLess(gradient @ direction, 0.0)

    def test_metric_rejects_nonsymmetric_and_indefinite_matrices(self):
        with self.assertRaisesRegex(ValueError, "symmetric"):
            validate_metric([[1.0, 0.5], [0.2, 1.0]], 2)
        with self.assertRaisesRegex(ValueError, "positive definite"):
            validate_metric([[1.0, 0.0], [0.0, -1.0]], 2)

    def test_generalized_spectrum_and_condition_number(self):
        hessian = np.diag([8.0, 2.0])
        identity = np.eye(2)
        values, vectors = generalized_spectrum(hessian, identity)
        np.testing.assert_allclose(values, [2.0, 8.0])
        np.testing.assert_allclose(hessian @ vectors, identity @ vectors @ np.diag(values))
        np.testing.assert_allclose(vectors.T @ identity @ vectors, identity)
        self.assertAlmostEqual(metric_condition_number(hessian, identity), 4.0)
        self.assertAlmostEqual(metric_condition_number(hessian, hessian), 1.0)

    def test_generalized_eigenvectors_are_metric_orthonormal(self):
        metric = np.array([[2.0, 0.3], [0.3, 1.0]])
        values, vectors = generalized_spectrum(self.H, metric)
        np.testing.assert_allclose(self.H @ vectors, metric @ vectors @ np.diag(values))
        np.testing.assert_allclose(vectors.T @ metric @ vectors, np.eye(2), atol=1e-12)

    def test_armijo_rejects_large_step_then_accepts_first_suitable_step(self):
        objective = Quadratic([[2.0]], [0.0])
        search = armijo_backtracking(
            objective, [1.0], [-2.0], alpha_0=1.0, beta=0.5, sigma=0.1
        )
        self.assertTrue(search.accepted)
        self.assertAlmostEqual(search.step, 0.5)
        self.assertEqual(search.trials.tolist(), [1.0, 0.5])
        self.assertGreater(search.values[0], search.bounds[0])
        self.assertLessEqual(search.values[1], search.bounds[1])

    def test_armijo_reports_backtracking_limit_without_success_fallback(self):
        objective = Quadratic([[2.0]], [0.0])
        search = armijo_backtracking(
            objective, [1.0], [-2.0], alpha_0=1.0, max_backtracks=0
        )
        self.assertFalse(search.accepted)
        self.assertIsNone(search.step)
        with self.assertRaisesRegex(ValueError, "descent direction"):
            armijo_backtracking(objective, [1.0], [1.0])

    def test_failed_armijo_search_is_not_counted_as_an_accepted_iteration(self):
        objective = Quadratic([[2.0]], [0.0])
        result = optimize(
            objective, [1.0], [[1.0]], alpha_0=1.0, max_backtracks=0
        )
        self.assertFalse(result.converged)
        self.assertFalse(result.iterations[0].armijo.accepted)
        self.assertEqual(metric_summary(objective, [[1.0]], result)["iterations"], 0.0)

    def test_metric_gradient_descent_converges_and_exact_metric_is_one_step(self):
        objective = Quadratic(np.diag([8.0, 2.0]), [-4.0, 2.0])
        result = optimize(objective, [3.0, -4.0], objective.H)
        self.assertTrue(result.converged)
        self.assertEqual(len(result.iterations) - 1, 1)
        np.testing.assert_allclose(result.points[-1], objective.minimizer, atol=1e-12)
        self.assertAlmostEqual(result.final_value, objective.minimum)

    def test_quadratic_and_hessian_require_positive_definiteness(self):
        with self.assertRaisesRegex(ValueError, "positive definite"):
            Quadratic([[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0])
        np.testing.assert_allclose(self.objective.hessian(), self.H)

    def test_plot_gallery_creates_separate_zoomable_dashboard_figures(self):
        laboratory = QuadraticLaboratory()
        figures = laboratory.create_plot_gallery()
        try:
            self.assertEqual(
                set(figures),
                {"contours", "surface", "armijo", "spectrum", "summary"},
            )
            for figure in figures.values():
                self.assertEqual(len(figure.axes), 1)
        finally:
            for figure in figures.values():
                plt.close(figure)
            plt.close(laboratory.fig)
            plt.close(laboratory.fig_contour)
            plt.close(laboratory.fig_surface)

    def test_interactive_views_are_separate_and_inner_product_is_editable(self):
        laboratory = QuadraticLaboratory()
        try:
            self.assertIsNot(laboratory.fig_contour, laboratory.fig_surface)
            self.assertIsNot(laboratory.fig_contour, laboratory.fig)
            self.assertIsNot(laboratory.fig_surface, laboratory.fig)
            self.assertIs(laboratory.ax_contour.figure, laboratory.fig_contour)
            self.assertIs(laboratory.ax_surface.figure, laboratory.fig_surface)
            self.assertEqual(len(laboratory.fig_contour.axes), 1)
            self.assertEqual(len(laboratory.fig_surface.axes), 1)

            inner_product = np.array([[3.0, 0.25], [0.25, 1.5]])
            for box, value in zip(laboratory._textbox_widgets[6:9], (3.0, 0.25, 1.5)):
                box.set_val(str(value))
            laboratory._apply_textboxes()

            np.testing.assert_allclose(laboratory.M, inner_product)
            np.testing.assert_allclose(laboratory.metrics[1], inner_product)
        finally:
            plt.close(laboratory.fig)
            plt.close(laboratory.fig_contour)
            plt.close(laboratory.fig_surface)


if __name__ == "__main__":
    unittest.main()
