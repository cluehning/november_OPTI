"""Matplotlib-Ansicht zum Vergleich verschiedener Metriken bei Quadraten."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider, TextBox
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

COLORS = ("#64b5f6", "#ffb74d", "#81c784")
METRIC_NAMES = ("Euclidean", "User metric", "Preconditioned")


def _matrix(value: NDArray | list[list[float]], name: str) -> FloatArray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if result.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(result, result.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be symmetric.")
    return (result + result.T) / 2


@dataclass(frozen=True)
class Quadratic:
    """Die streng konvexe quadratische Funktion 1/2 x.T H x + b.T x + c."""

    H: FloatArray
    b: FloatArray
    c: float = 0.0

    def __post_init__(self) -> None:
        hessian = _matrix(self.H, "H")
        vector = np.asarray(self.b, dtype=float)
        if vector.shape != (hessian.shape[0],) or not np.all(np.isfinite(vector)):
            raise ValueError("b must be a finite vector matching H.")
        if not np.isfinite(self.c):
            raise ValueError("c must be finite.")
        if np.linalg.eigvalsh(hessian)[0] <= 0:
            raise ValueError("H must be positive definite.")
        object.__setattr__(self, "H", hessian)
        object.__setattr__(self, "b", vector)
        object.__setattr__(self, "c", float(self.c))

    @property
    def minimizer(self) -> FloatArray:
        return np.linalg.solve(self.H, -self.b)

    @property
    def minimum(self) -> float:
        return self.value(self.minimizer)

    def value(self, x: NDArray | list[float]) -> float:
        point = np.asarray(x, dtype=float)
        return float(0.5 * point @ self.H @ point + self.b @ point + self.c)

    def gradient(self, x: NDArray | list[float]) -> FloatArray:
        return self.H @ np.asarray(x, dtype=float) + self.b

    def hessian(self) -> FloatArray:
        return self.H.copy()


def validate_metric(M: NDArray | list[list[float]], dimension: int) -> FloatArray:
    """Gib eine symmetrische, positiv definite Matrix des Innenprodukts zurück."""
    metric = _matrix(M, "M")
    if metric.shape != (dimension, dimension):
        raise ValueError("M must have the same dimensions as H.")
    try:
        np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as error:
        raise ValueError("M must be positive definite.") from error
    return metric


def metric_norm(x: NDArray | list[float], M: NDArray | list[list[float]]) -> float:
    """Berechne die M-Norm ||x||_M = sqrt(x.T M x)."""
    vector = np.asarray(x, dtype=float)
    metric = validate_metric(M, vector.size)
    return float(np.sqrt(max(0.0, vector @ metric @ vector)))


def metric_gradient(
    gradient: NDArray | list[float], M: NDArray | list[list[float]]
) -> FloatArray:
    """Gib den M-Gradienten grad_M f = M^{-1} grad f zurück."""
    vector = np.asarray(gradient, dtype=float)
    metric = validate_metric(M, vector.size)
    return np.linalg.solve(metric, vector)


def steepest_descent_direction(
    gradient: NDArray | list[float], M: NDArray | list[list[float]]
) -> FloatArray:
    """Gib die steilste Abstiegsrichtung in der M-Metrik zurück."""
    metric_gradient_value = metric_gradient(gradient, M)
    norm = metric_norm(metric_gradient_value, M)
    if norm == 0.0:
        return np.zeros_like(metric_gradient_value)
    return -metric_gradient_value / norm


def generalized_spectrum(
    H: NDArray | list[list[float]], M: NDArray | list[list[float]]
) -> tuple[FloatArray, FloatArray]:
    """Gib Eigenpaare von H v = lambda M v mit M-normierten Eigenvektoren zurück."""
    hessian = _matrix(H, "H")
    metric = validate_metric(M, hessian.shape[0])
    factor = np.linalg.cholesky(metric)
    whitened = np.linalg.solve(factor, hessian)
    whitened = np.linalg.solve(factor, whitened.T).T
    eigenvalues, orthogonal_vectors = np.linalg.eigh(whitened)
    eigenvectors = np.linalg.solve(factor.T, orthogonal_vectors)
    return eigenvalues, eigenvectors


def metric_condition_number(
    H: NDArray | list[list[float]], M: NDArray | list[list[float]]
) -> float:
    """Berechne die spektrale Konditionszahl von M^{-1/2} H M^{-1/2}."""
    eigenvalues, _ = generalized_spectrum(H, M)
    if eigenvalues[0] <= 0:
        raise ValueError("H must be positive definite to have this condition number.")
    return float(eigenvalues[-1] / eigenvalues[0])


@dataclass(frozen=True)
class ArmijoResult:
    """Protokoll der Rückwärtssuche, auch wenn sie das Limit erreicht."""

    step: float | None
    trials: FloatArray
    values: FloatArray
    bounds: FloatArray
    accepted: bool


def armijo_backtracking(
    objective: Quadratic,
    x: NDArray | list[float],
    direction: NDArray | list[float],
    *,
    alpha_0: float = 1.0,
    beta: float = 0.5,
    sigma: float = 0.1,
    max_backtracks: int = 30,
) -> ArmijoResult:
    """Finde die erste geometrische Schrittweite, die Armijo erfüllt."""
    if not np.isfinite(alpha_0) or alpha_0 <= 0:
        raise ValueError("alpha_0 must be positive and finite.")
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta must be strictly between zero and one.")
    if not np.isfinite(sigma) or not 0 < sigma < 1:
        raise ValueError("sigma must be strictly between zero and one.")
    if not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer.")
    point = np.asarray(x, dtype=float)
    vector = np.asarray(direction, dtype=float)
    if point.shape != objective.b.shape or vector.shape != point.shape:
        raise ValueError("x and direction must match the objective dimension.")
    slope = float(objective.gradient(point) @ vector)
    if slope >= 0:
        raise ValueError("The search direction must be a strict descent direction.")

    trials: list[float] = []
    values: list[float] = []
    bounds: list[float] = []
    value_at_x = objective.value(point)
    for index in range(max_backtracks + 1):
        step = alpha_0 * beta**index
        trial_value = objective.value(point + step * vector)
        armijo_bound = value_at_x + sigma * step * slope
        trials.append(step)
        values.append(trial_value)
        bounds.append(armijo_bound)
        if trial_value <= armijo_bound:
            return ArmijoResult(
                step,
                np.asarray(trials),
                np.asarray(values),
                np.asarray(bounds),
                True,
            )
    return ArmijoResult(
        None,
        np.asarray(trials),
        np.asarray(values),
        np.asarray(bounds),
        False,
    )


@dataclass(frozen=True)
class Iteration:
    x: FloatArray
    value: float
    gradient: FloatArray
    direction: FloatArray
    directional_derivative: float
    rayleigh_quotient: float
    armijo: ArmijoResult | None


@dataclass(frozen=True)
class OptimizationResult:
    metric: FloatArray
    iterations: tuple[Iteration, ...]
    converged: bool
    message: str

    @property
    def points(self) -> FloatArray:
        return np.stack([iteration.x for iteration in self.iterations])

    @property
    def final_value(self) -> float:
        return self.iterations[-1].value


def optimize(
    objective: Quadratic,
    x0: NDArray | list[float],
    M: NDArray | list[list[float]],
    *,
    alpha_0: float = 1.0,
    beta: float = 0.5,
    sigma: float = 0.1,
    max_backtracks: int = 30,
    max_iterations: int = 100,
    tolerance: float = 1e-8,
) -> OptimizationResult:
    """Führe steilsten Abstieg in einer Metrik mit Armijo-Rückwärtssuche aus."""
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError("max_iterations must be a nonnegative integer.")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be positive and finite.")
    point = np.asarray(x0, dtype=float).copy()
    if point.shape != objective.b.shape or not np.all(np.isfinite(point)):
        raise ValueError("x0 must be a finite vector matching the objective.")
    metric = validate_metric(M, point.size)
    entries: list[Iteration] = []
    for index in range(max_iterations + 1):
        gradient = objective.gradient(point)
        if np.linalg.norm(gradient) <= tolerance:
            entries.append(Iteration(point.copy(), objective.value(point), gradient,
                                     np.zeros_like(point), 0.0, float("nan"), None))
            return OptimizationResult(metric, tuple(entries), True, "Gradient tolerance reached.")
        if index == max_iterations:
            break
        direction = -metric_gradient(gradient, metric)
        slope = float(gradient @ direction)
        rayleigh = float(direction @ objective.H @ direction / (direction @ metric @ direction))
        search = armijo_backtracking(
            objective,
            point,
            direction,
            alpha_0=alpha_0,
            beta=beta,
            sigma=sigma,
            max_backtracks=max_backtracks,
        )
        entries.append(
            Iteration(point.copy(), objective.value(point), gradient.copy(),
                      direction.copy(), slope, rayleigh, search)
        )
        if not search.accepted:
            return OptimizationResult(metric, tuple(entries), False,
                                      "Armijo search failed at its backtracking limit.")
        point = point + search.step * direction

    gradient = objective.gradient(point)
    entries.append(Iteration(point.copy(), objective.value(point), gradient,
                             np.zeros_like(point), 0.0, float("nan"), None))
    converged = np.linalg.norm(gradient) <= tolerance
    message = "Gradient tolerance reached." if converged else "Maximum iteration count reached."
    return OptimizationResult(metric, tuple(entries), converged, message)


def metric_summary(
    objective: Quadratic, M: NDArray, result: OptimizationResult
) -> dict[str, float]:
    """Fasse verallgemeinerte Krümmung und beobachtete Konvergenz zusammen."""
    eigenvalues, _ = generalized_spectrum(objective.H, M)
    condition = float(eigenvalues[-1] / eigenvalues[0])
    exact_line_search_factor = ((condition - 1.0) / (condition + 1.0)) ** 2
    return {
        "lambda_min": float(eigenvalues[0]),
        "lambda_max": float(eigenvalues[-1]),
        "condition": condition,
        "exact_line_search_factor": exact_line_search_factor,
        "iterations": float(sum(
            entry.armijo is not None and entry.armijo.accepted
            for entry in result.iterations
        )),
        "final_value": result.final_value,
    }


class QuadraticLaboratory:
    """Interaktive Matplotlib-Ansicht zum quadratischen Metrikvergleich."""

    def __init__(
        self,
        H: NDArray | None = None,
        b: NDArray | None = None,
        c: float = 0.0,
        x0: NDArray | None = None,
        M: NDArray | None = None,
        *,
        alpha_0: float = 1.0,
        beta: float = 0.5,
        sigma: float = 0.1,
        max_backtracks: int = 30,
        max_iterations: int = 80,
        plot_range: float = 5.0,
        interval_ms: int = 500,
    ) -> None:
        self.H = np.asarray(H if H is not None else [[9.0, 3.0], [3.0, 2.0]], dtype=float)
        self.b = np.asarray(b if b is not None else [-2.0, 1.0], dtype=float)
        self.c = float(c)
        self.x0 = np.asarray(x0 if x0 is not None else [-3.0, 3.0], dtype=float)
        self.M = np.asarray(M if M is not None else [[2.0, 0.55], [0.55, 1.0]], dtype=float)
        self.alpha_0, self.beta, self.sigma = float(alpha_0), float(beta), float(sigma)
        self.max_backtracks = int(max_backtracks)
        self.max_iterations = int(max_iterations)
        self.plot_range = float(plot_range)
        self.interval_ms = int(interval_ms)
        self.metric_index = 0
        self.iteration_index = 0
        self.theme = "dark"
        self.visible = {
            "levels": True, "minimizer": True, "iterate": True,
            "gradient": True, "direction": True, "tangent": True,
            "trials": True, "iterations": True, "trajectory": True,
        }
        self._initial_artist = None
        self._picked_paths: dict[int, tuple[int, FloatArray]] = {}
        self._dragging = False
        self._path_picked = False
        self._contour_view_initialized = False
        self._reset_contour_view = True
        self._surface_view = (29, -53)
        self._timer = None
        self._build_figure()
        self._recompute()
        self._render()

    def _objective(self) -> Quadratic:
        if self.H.shape != (2, 2) or self.b.shape != (2,) or self.x0.shape != (2,):
            raise ValueError("The interactive lab is two-dimensional.")
        return Quadratic(self.H, self.b, self.c)

    def _metrics(self) -> tuple[FloatArray, ...]:
        objective = self._objective()
        user_metric = validate_metric(self.M, 2)
        preconditioner = np.diag(np.diag(objective.H))
        validate_metric(preconditioner, 2)
        return np.eye(2), user_metric, preconditioner

    def _recompute(self) -> None:
        self.objective = self._objective()
        self.metrics = self._metrics()
        settings = {
            "alpha_0": self.alpha_0,
            "beta": self.beta,
            "sigma": self.sigma,
            "max_backtracks": self.max_backtracks,
            "max_iterations": self.max_iterations,
        }
        self.results = tuple(
            optimize(self.objective, self.x0, metric, **settings)
            for metric in self.metrics
        )
        self.summaries = tuple(
            metric_summary(self.objective, metric, result)
            for metric, result in zip(self.metrics, self.results)
        )
        self.iteration_index = min(
            self.iteration_index,
            max(0, len(self.results[self.metric_index].iterations) - 1),
        )

    def _build_figure(self) -> None:
        self.fig_contour = plt.figure(figsize=(9, 7), facecolor="#101722")
        self.fig_contour.suptitle(
            "CONTOUR SETS  /  METRIC TRAJECTORIES",
            color="#e7edf5", fontsize=13, fontweight="bold",
        )
        self.ax_contour = self.fig_contour.add_subplot(111)

        self.fig_surface = plt.figure(figsize=(10, 7), facecolor="#101722")
        self.fig_surface.suptitle(
            "3D OBJECTIVE SURFACE  /  METRIC TRAJECTORIES",
            color="#e7edf5", fontsize=13, fontweight="bold",
        )
        self.ax_surface = self.fig_surface.add_subplot(111, projection="3d")

        self.fig = plt.figure(figsize=(17, 11), facecolor="#101722")
        self.fig.suptitle(
            "VARIABLES & CONTROLS  /  INNER PRODUCT M · ARMIJO · METRICS",
            color="#e7edf5", fontsize=14, fontweight="bold", y=0.98,
        )
        grid = self.fig.add_gridspec(
            2, 3, left=0.055, right=0.98, top=0.92, bottom=0.37,
            height_ratios=(1.05, 0.95), hspace=0.36, wspace=0.28,
        )
        self.ax_armijo = self.fig.add_subplot(grid[0, 0])
        self.ax_explanation = self.fig.add_subplot(grid[0, 2])
        spectrum_grid = grid[1, :2].subgridspec(2, 1, height_ratios=(1, 1.15), hspace=0.2)
        self.ax_eigenvalues = self.fig.add_subplot(spectrum_grid[0, 0])
        self.ax_summary = self.fig.add_subplot(spectrum_grid[1, 0])
        self.ax_explanation.axis("off")
        self._textbox_widgets: list[TextBox] = []
        self._create_textboxes()
        self._create_buttons()
        for figure in (self.fig, self.fig_contour, self.fig_surface):
            figure.canvas.mpl_connect("pick_event", self._on_pick)
            figure.canvas.mpl_connect("key_press_event", self._on_key)
        for figure in (self.fig, self.fig_contour):
            figure.canvas.mpl_connect("button_press_event", self._on_press)
            figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
            figure.canvas.mpl_connect("button_release_event", self._on_release)
        self._timer = self.fig.canvas.new_timer(interval=self.interval_ms)
        self._timer.add_callback(self._advance_animation)

    def _create_textboxes(self) -> None:
        rows = [
            (0.293, [("H11", self.H[0, 0]), ("H12", self.H[0, 1]),
                     ("H22", self.H[1, 1]), ("b1", self.b[0]), ("b2", self.b[1]),
                     ("c", self.c)]),
            (0.248, [("M11", self.M[0, 0]), ("M12", self.M[0, 1]),
                     ("M22", self.M[1, 1]), ("x01", self.x0[0]), ("x02", self.x0[1]),
                     ("Range", self.plot_range)]),
            (0.203, [("alpha0", self.alpha_0), ("beta", self.beta),
                     ("sigma", self.sigma), ("BT max", self.max_backtracks),
                     ("Max iter", self.max_iterations), ("ms", self.interval_ms)]),
        ]
        width, gap = 0.112, 0.023
        for y, fields in rows:
            for index, (label, value) in enumerate(fields):
                x = 0.06 + index * (width + gap)
                axes = self.fig.add_axes([x, y, width, 0.032], facecolor="#202c3a")
                box = TextBox(
                    axes, label, initial=f"{value:g}",
                    color="#202c3a", hovercolor="#2b3c50",
                    label_pad=0.05, textalignment="center",
                )
                box.label.set_color("#a9bacb")
                box.text_disp.set_color("#eef3f8")
                self._textbox_widgets.append(box)
        for box in self._textbox_widgets:
            box.on_submit(self._apply_textboxes)
        self._control_caption = self.fig.text(
            0.06, 0.33,
            "INNER PRODUCT: edit M11, M12, M22 (M12 is mirrored; M must be SPD)   •   "
            "H12 is mirrored   •   press Enter or RECOMPUTE to apply",
            color="#91a2b5", fontsize=8,
        )

    def _create_buttons(self) -> None:
        actions: tuple[tuple[str, Callable], ...] = (
            ("◀ STEP", self._step_back),
            ("STEP ▶", self._step_forward),
            ("PLAY", self._toggle_play),
            ("RESET", self._reset),
            ("LIGHT / DARK", self._toggle_theme),
            ("RECOMPUTE", self._apply_textboxes),
        )
        for index, (label, callback) in enumerate(actions):
            axes = self.fig.add_axes([0.06 + index * 0.106, 0.135, 0.092, 0.038])
            button = Button(axes, label, color="#26364a", hovercolor="#3d5470")
            button.label.set_color("#edf2f8")
            button.on_clicked(callback)
            setattr(self, f"_button_{index}", button)
        axes = self.fig.add_axes([0.72, 0.05, 0.125, 0.11], facecolor="#182331")
        self._metric_radio = RadioButtons(axes, METRIC_NAMES, active=0, activecolor="#64b5f6")
        for label in self._metric_radio.labels:
            label.set_color("#d9e2ed")
            label.set_fontsize(8)
        self._metric_radio.on_clicked(self._select_metric)
        axes = self.fig.add_axes([0.85, 0.03, 0.145, 0.15], facecolor="#182331")
        labels = (
            "level sets", "minimizer", "current point", "trajectory",
            "gradient", "direction", "tangent", "trials", "iterations",
        )
        self._visibility_keys = {
            "level sets": "levels",
            "minimizer": "minimizer",
            "current point": "iterate",
            "trajectory": "trajectory",
            "gradient": "gradient",
            "direction": "direction",
            "tangent": "tangent",
            "trials": "trials",
            "iterations": "iterations",
        }
        self._checks = CheckButtons(
            axes, labels, [self.visible[self._visibility_keys[label]] for label in labels]
        )
        for label in self._checks.labels:
            label.set_color("#d9e2ed")
            label.set_fontsize(8)
        self._checks.on_clicked(self._toggle_visibility)
        slider_ax = self.fig.add_axes([0.06, 0.075, 0.42, 0.018], facecolor="#202c3a")
        self._iteration_slider = Slider(
            slider_ax, "Iteration", 0, 1, valinit=0, valstep=1,
            color="#64b5f6", track_color="#30445b",
        )
        self._iteration_slider.label.set_color("#c4d1df")
        self._iteration_slider.valtext.set_color("#eef3f8")
        self._iteration_slider.on_changed(self._select_iteration)
        self._key_help = self.fig.text(
            0.49, 0.08,
            "← / → step · Space play/pause\n"
            "click/drag contour to set start · pick path point",
            color="#91a2b5", fontsize=7,
        )

    def _style_axes(self) -> None:
        dark = self.theme == "dark"
        background = "#101722" if dark else "#f7f9fc"
        foreground = "#d8e2ed" if dark else "#263547"
        panel = "#141e2b" if dark else "#ffffff"
        self.fig.set_facecolor(background)
        self.fig_contour.set_facecolor(background)
        self.fig_surface.set_facecolor(background)
        for ax in (
            self.ax_contour, self.ax_armijo, self.ax_eigenvalues,
            self.ax_summary, self.ax_explanation,
        ):
            ax.set_facecolor(panel)
            ax.tick_params(colors=foreground, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color("#344357" if dark else "#cbd5e1")
            ax.xaxis.label.set_color(foreground)
            ax.yaxis.label.set_color(foreground)
            ax.title.set_color(foreground)
        self.ax_surface.set_facecolor(panel)
        self.ax_surface.tick_params(colors=foreground, labelsize=7)
        self.ax_surface.xaxis.label.set_color(foreground)
        self.ax_surface.yaxis.label.set_color(foreground)
        self.ax_surface.zaxis.label.set_color(foreground)
        pane_color = (*colors.to_rgba(panel)[:3], 1.0)
        for axis in (self.ax_surface.xaxis, self.ax_surface.yaxis, self.ax_surface.zaxis):
            axis.pane.set_facecolor(pane_color)
        for figure in (self.fig, self.fig_contour, self.fig_surface):
            figure._suptitle.set_color(foreground)
        for box in self._textbox_widgets:
            box.ax.set_facecolor(panel)
            box.color = panel
            box.hovercolor = "#2b3c50" if dark else "#e8eef5"
            box.text_disp.set_color(foreground)
            box.label.set_color(foreground)
        for index in range(6):
            button = getattr(self, f"_button_{index}")
            button.ax.set_facecolor(panel)
            button.color = panel
            button.hovercolor = "#3d5470" if dark else "#e3eaf2"
            button.label.set_color(foreground)
        self._metric_radio.ax.set_facecolor(panel)
        self._checks.ax.set_facecolor(panel)
        for label in (*self._metric_radio.labels, *self._checks.labels):
            label.set_color(foreground)
        self._iteration_slider.ax.set_facecolor(panel)
        self._iteration_slider.label.set_color(foreground)
        self._iteration_slider.valtext.set_color(foreground)
        self._control_caption.set_color(foreground)
        self._key_help.set_color(foreground)
        self.fig._lab_foreground = foreground

    def _render(self) -> None:
        self._draw_contours()
        self._draw_surface()
        self._draw_armijo()
        self._draw_spectrum()
        self._draw_explanation()
        self._style_axes()
        for figure in (self.fig, self.fig_contour, self.fig_surface):
            figure.canvas.draw_idle()

    def _draw_contours(self) -> None:
        ax, objective = self.ax_contour, self.objective
        previous_xlim, previous_ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
        self._picked_paths.clear()
        xstar = objective.minimizer
        extent = self.plot_range
        x = np.linspace(xstar[0] - extent, xstar[0] + extent, 100)
        y = np.linspace(xstar[1] - extent, xstar[1] + extent, 100)
        xx, yy = np.meshgrid(x, y)
        values = (
            0.5 * (
                objective.H[0, 0] * xx**2
                + 2 * objective.H[0, 1] * xx * yy
                + objective.H[1, 1] * yy**2
            )
            + objective.b[0] * xx + objective.b[1] * yy + objective.c
        )
        if self.visible["levels"]:
            positive_levels = np.geomspace(
                max((float(values.max()) - objective.minimum) * 1e-5, 1e-9),
                max(float(values.max()) - objective.minimum, 1e-8),
                24,
            )
            ax.contour(xx, yy, values, levels=objective.minimum + positive_levels,
                       colors="#41566e", linewidths=0.75, alpha=0.9)
        for index, (name, result) in enumerate(zip(METRIC_NAMES, self.results)):
            points = result.points
            eigenvalues, eigenvectors = np.linalg.eigh(self.metrics[index])
            angles = np.linspace(0, 2 * np.pi, 120)
            radius = min(self.plot_range * 0.12, 0.8)
            ellipse = self.objective.minimizer[:, None] + eigenvectors @ (
                radius / np.sqrt(eigenvalues)[:, None]
                * np.vstack((np.cos(angles), np.sin(angles)))
            )
            ax.plot(ellipse[0], ellipse[1], color=COLORS[index], alpha=0.7,
                    linestyle=":", linewidth=1.0, label="_nolegend_")
            if self.visible["trajectory"]:
                path, = ax.plot(points[:, 0], points[:, 1], "-o", color=COLORS[index],
                                linewidth=1.5, markersize=3.5, label=name,
                                alpha=0.9, picker=5)
                self._picked_paths[id(path)] = (index, points)
                for iteration, point in enumerate(points):
                    if self.visible["iterations"] and iteration % max(1, len(points) // 10) == 0:
                        ax.annotate(str(iteration), point, xytext=(3, 4),
                                    textcoords="offset points", color=COLORS[index],
                                    fontsize=6, alpha=0.8)
            path_index = min(self.iteration_index, len(points) - 1)
            selected_point = points[path_index]
            if index == self.metric_index:
                if self.visible["iterate"]:
                    ax.scatter(*selected_point, s=95, color="#ffffff", edgecolor=COLORS[index],
                               linewidth=2, zorder=6)
                self._initial_artist = ax.scatter(
                    *points[0], s=95, marker="D", color=COLORS[index],
                    edgecolor="#ffffff", linewidth=1, zorder=7, picker=8,
                )
                current = result.iterations[path_index]
                self._draw_selected_vectors(ax, current, objective)
            else:
                ax.scatter(*selected_point, s=42, color=COLORS[index], edgecolor="#111820",
                           linewidth=1, zorder=5)
        if self.visible["minimizer"]:
            ax.scatter(*xstar, marker="*", s=180, color="#ef6c8e",
                       edgecolor="#ffffff", linewidth=0.8, label="Minimizer", zorder=8)
        if self._contour_view_initialized and not self._reset_contour_view:
            ax.set_xlim(previous_xlim)
            ax.set_ylim(previous_ylim)
        else:
            ax.set_xlim(xstar[0] - extent, xstar[0] + extent)
            ax.set_ylim(xstar[1] - extent, xstar[1] + extent)
        self._contour_view_initialized = True
        self._reset_contour_view = False
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"Level sets & metric trajectories  •  f* = {objective.minimum:.4g}", fontsize=9)
        ax.text(
            0.02, 0.02,
            "Dotted ellipses: equal metric-norm circles; line colors identify each metric.",
            transform=ax.transAxes, color="#91a2b5", fontsize=6, va="bottom",
        )
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        ax.legend(fontsize=7, facecolor="#1b2735", labelcolor="#e1e9f2",
                  edgecolor="#344357", loc="upper right")

    def _draw_selected_vectors(self, ax: Axes, current: Iteration, objective: Quadratic) -> None:
        point, gradient = current.x, current.gradient
        scale = self.plot_range / max(np.linalg.norm(gradient), np.linalg.norm(current.direction), 1.0)
        if self.visible["gradient"]:
            ax.quiver(*point, *(gradient * scale), color="#f48fb1",
                      angles="xy", scale_units="xy", scale=1, width=0.008,
                      label="_nolegend_")
        if self.visible["direction"]:
            ax.quiver(*point, *(current.direction * scale * 2.5), color="#fff176",
                      angles="xy", scale_units="xy", scale=1, width=0.008,
                      label="_nolegend_")
        if self.visible["tangent"] and current.armijo is not None:
            span = self.plot_range * 0.45
            axis = np.linspace(-span, span, 60)
            if abs(gradient[1]) > 1e-10:
                ax.plot(point[0] + axis, point[1] - gradient[0] / gradient[1] * axis,
                        "--", color="#ce93d8", linewidth=1, label="_nolegend_")
            elif abs(gradient[0]) > 1e-10:
                ax.axvline(point[0], color="#ce93d8", linestyle="--",
                           linewidth=1, label="_nolegend_")
        if self.visible["trials"] and current.armijo is not None:
            search = current.armijo
            trial_points = point + search.trials[:, None] * current.direction
            rejected = search.values > search.bounds
            if np.any(rejected):
                ax.scatter(trial_points[rejected, 0], trial_points[rejected, 1],
                           marker="x", color="#ef6c8e", s=52,
                           label="_nolegend_", zorder=8)
            if search.accepted:
                accepted = trial_points[-1]
                ax.scatter(*accepted, marker="*", color="#a5d6a7", s=120,
                           edgecolor="#ffffff", linewidth=0.6,
                           label="_nolegend_", zorder=9)

    def _draw_surface(self) -> None:
        ax, objective = self.ax_surface, self.objective
        if hasattr(ax, "elev") and self._surface_view is not None:
            self._surface_view = (ax.elev, ax.azim)
        ax.clear()
        xstar = objective.minimizer
        extent = self.plot_range
        x = np.linspace(xstar[0] - extent, xstar[0] + extent, 42)
        y = np.linspace(xstar[1] - extent, xstar[1] + extent, 42)
        xx, yy = np.meshgrid(x, y)
        zz = (
            0.5 * (objective.H[0, 0] * xx**2 + 2 * objective.H[0, 1] * xx * yy
                   + objective.H[1, 1] * yy**2)
            + objective.b[0] * xx + objective.b[1] * yy + objective.c
        )
        ax.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.48, linewidth=0, antialiased=True)
        for index, result in enumerate(self.results):
            points = result.points
            values = np.array([objective.value(point) for point in points])
            if self.visible["trajectory"]:
                ax.plot(points[:, 0], points[:, 1], values, color=COLORS[index],
                        marker="o", markersize=3, linewidth=1.8, label=METRIC_NAMES[index])
                artist = ax.scatter(points[:, 0], points[:, 1], values,
                                    color=COLORS[index], s=25, picker=True, depthshade=False)
                self._picked_paths[id(artist)] = (index, points)
            point = points[min(self.iteration_index, len(points) - 1)]
            if index == self.metric_index:
                if self.visible["iterate"]:
                    ax.scatter(*point, objective.value(point), color="#ffffff", s=45,
                               edgecolor=COLORS[index], linewidth=1.5, depthshade=False)
                current = result.iterations[min(self.iteration_index, len(points) - 1)]
                if self.visible["gradient"] and np.linalg.norm(current.gradient) > 1e-12:
                    vector = current.gradient
                    step = 0.12 * extent / np.linalg.norm(vector)
                    ax.quiver(
                        current.x[0], current.x[1], current.value,
                        step * vector[0], step * vector[1],
                        step * float(vector @ vector), color="#f48fb1",
                        arrow_length_ratio=0.3, linewidth=1.4,
                    )
        if self.visible["minimizer"]:
            ax.scatter(*xstar, objective.minimum, marker="*", s=110, color="#ef6c8e",
                       edgecolor="#ffffff", linewidth=0.7, depthshade=False)
        ax.set_title("Synchronized objective surface & paths", fontsize=9)
        ax.set_xlabel("x₁", labelpad=1)
        ax.set_ylabel("x₂", labelpad=1)
        ax.set_zlabel("f(x)", labelpad=0)
        ax.view_init(elev=self._surface_view[0], azim=self._surface_view[1])

    def _draw_armijo(self) -> None:
        ax = self.ax_armijo
        ax.clear()
        result = self.results[self.metric_index]
        current = result.iterations[min(self.iteration_index, len(result.iterations) - 1)]
        if current.armijo is None:
            ax.text(0.5, 0.5, "At a stationary point\n(no line search needed)",
                    ha="center", va="center", color="#b7c7d8", transform=ax.transAxes)
            ax.set_title("Armijo backtracking", fontsize=9)
            return
        search = current.armijo
        upper = search.trials[0]
        lower = search.trials[-1] * self.beta
        steps = np.geomspace(max(lower, 1e-12), upper, 180)
        objective_values = np.array([
            self.objective.value(current.x + step * current.direction) for step in steps
        ])
        bound_values = (
            current.value + self.sigma * steps * current.directional_derivative
        )
        ax.plot(steps, objective_values, color="#64b5f6", linewidth=2, label="f(xₖ + t dₖ)")
        ax.plot(steps, bound_values, "--", color="#ffb74d", linewidth=1.7,
                label="f(xₖ) + σt∇fᵀd")
        for step, value, bound in zip(search.trials, search.values, search.bounds):
            accepted = value <= bound
            ax.scatter(step, value, marker="o" if accepted else "x",
                       color="#81c784" if accepted else "#ef6c8e", s=58, zorder=5)
        if search.accepted:
            ax.axvline(search.step, color="#81c784", alpha=0.6, linestyle=":")
            ax.annotate(f"first accepted t={search.step:.3g}",
                        (search.step, search.values[-1]), xytext=(5, 10),
                        textcoords="offset points", color="#81c784", fontsize=7)
        else:
            ax.text(0.04, 0.96, "No acceptable step within the backtracking limit",
                    transform=ax.transAxes, va="top", color="#ef9a9a", fontsize=7)
        ax.set_xscale("log")
        ax.set_title(
            f"Armijo  •  α₀={self.alpha_0:g}, β={self.beta:g}, σ={self.sigma:g}",
            fontsize=9,
        )
        ax.set_xlabel("step t (log scale)")
        ax.set_ylabel("objective / sufficient-decrease bound")
        ax.legend(fontsize=7, facecolor="#1b2735", labelcolor="#e1e9f2", edgecolor="#344357")

    def _draw_spectrum(self) -> None:
        ax = self.ax_eigenvalues
        table_ax = self.ax_summary
        ax.clear()
        table_ax.clear()
        rows = np.arange(len(METRIC_NAMES))
        for index, (name, metric, summary) in enumerate(
            zip(METRIC_NAMES, self.metrics, self.summaries)
        ):
            minimum, maximum = summary["lambda_min"], summary["lambda_max"]
            ax.plot([minimum, maximum], [index, index], color=COLORS[index],
                    linewidth=3, alpha=0.7)
            ax.scatter([minimum, maximum], [index, index], color=COLORS[index],
                       s=[28, 55], zorder=3)
            annotation_color = "#e1e9f2" if self.theme == "dark" else "#263547"
            ax.annotate(f"{minimum:.3g}", (minimum, index), xytext=(0, -10),
                        textcoords="offset points", ha="center", color=annotation_color,
                        fontsize=7)
            ax.annotate(f"{maximum:.3g}", (maximum, index), xytext=(0, 5),
                        textcoords="offset points", ha="center", color=annotation_color,
                        fontsize=7)
        ax.set_xscale("log")
        ax.set_yticks(rows, METRIC_NAMES)
        ax.set_title("Generalized eigenvalues  •  H v = λ M v", fontsize=8)
        ax.set_xlabel("curvature λ (log scale)")
        ax.grid(axis="x", color="#41566e", alpha=0.35)
        table_ax.axis("off")
        headers = [
            "Metric", "κ", "λmin", "λmax", "q* bound", "steps",
            "f final", "f − f*", "axes",
        ]
        table_data = []
        for name, summary in zip(METRIC_NAMES, self.summaries):
            axis_lengths = (
                1 / np.sqrt(summary["lambda_max"]),
                1 / np.sqrt(summary["lambda_min"]),
            )
            table_data.append([
                name,
                f"{summary['condition']:.3g}",
                f"{summary['lambda_min']:.3g}",
                f"{summary['lambda_max']:.3g}",
                f"{summary['exact_line_search_factor']:.3g}",
                str(int(summary["iterations"])),
                f"{summary['final_value']:.3g}",
                f"{summary['final_value'] - self.objective.minimum:.2g}",
                f"{axis_lengths[0]:.2g}, {axis_lengths[1]:.2g}",
            ])
        table = table_ax.table(
            cellText=table_data, colLabels=headers, cellLoc="center",
            colLoc="center", loc="center",
            colWidths=[0.18, 0.07, 0.08, 0.08, 0.1, 0.07, 0.15, 0.11, 0.12],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.2)
        for (row, _column), cell in table.get_celld().items():
            cell.set_edgecolor("#405168")
            cell.set_facecolor("#26374a" if row == 0 else
                               ("#172432" if row % 2 else "#1c2a39"))
            cell.get_text().set_color("#e1e9f2")
        table_ax.text(
            0.01, 0.04,
            "q* is the exact-line-search worst-case function-error factor (not an Armijo guarantee). "
            "Axis lengths are 1/√λ, in generalized eigen-coordinates.",
            transform=table_ax.transAxes,
            color="#aab9c9" if self.theme == "dark" else "#536273", fontsize=6.3,
            va="bottom",
        )

    def _draw_explanation(self) -> None:
        ax = self.ax_explanation
        ax.clear()
        ax.axis("off")
        result = self.results[self.metric_index]
        summary = self.summaries[self.metric_index]
        iteration = min(self.iteration_index, len(result.iterations) - 1)
        current = result.iterations[iteration]
        eigenvalues, eigenvectors = generalized_spectrum(
            self.objective.H, self.metrics[self.metric_index]
        )
        lines = [
            f"{METRIC_NAMES[self.metric_index].upper()} · ITERATION {iteration}",
            f"M = {np.array2string(self.metrics[self.metric_index], precision=2)}",
            f"κ = {summary['condition']:.3g}   λ = "
            f"[{eigenvalues[0]:.3g}, {eigenvalues[-1]:.3g}]",
            f"eigenvectors V = {np.array2string(eigenvectors, precision=2, suppress_small=True)}",
            f"axes 1/√λ = [{1 / np.sqrt(eigenvalues[-1]):.2g}, "
            f"{1 / np.sqrt(eigenvalues[0]):.2g}]",
            f"x* = {np.array2string(self.objective.minimizer, precision=2)}   "
            f"f(xₖ) = {current.value:.4g}",
            f"||∇f|| = {np.linalg.norm(current.gradient):.2g}   "
            f"||xₖ||ₘ = {metric_norm(current.x, self.metrics[self.metric_index]):.3g}",
            "",
            "GRADIENT  ·  WHY?",
            r"$\mathrm{d}f=\nabla f^\top\mathrm{d}x"
            r"=\langle M^{-1}\nabla f,\mathrm{d}x\rangle_M$",
            r"$\nabla_Mf=M^{-1}(Hx+b)$; H is the Euclidean Hessian.",
            "",
            "STEEPEST DESCENT  ·  WHY?",
            r"$d=-M^{-1}\nabla f$ minimizes $\nabla f^\top d$ over $\|d\|_M=1$.",
            fr"$\nabla f^\top d=-\|\nabla_M f\|_M^2$; Rayleigh curvature = "
            f"{current.rayleigh_quotient:.3g}"
            if np.isfinite(current.rayleigh_quotient)
            else "At the minimizer, the gradient and search direction vanish.",
        ]
        if current.armijo is not None:
            search = current.armijo
            lines.extend([
                "",
                "ARMIJO  ·  WHY ACCEPT OR REJECT?",
                (f"{len(search.trials)} trials; first acceptable t = {search.step:.3g}."
                 if search.accepted else f"{len(search.trials)} trials; no acceptable step."),
                r"$f(x+td)\leq f(x)+\sigma t\nabla f^\top d$.",
            ])
        lines.extend([
            "",
            "CONDITIONING  ·  WHY DOES IT MATTER?",
            "Large κ elongates level sets and makes descent zig-zag.",
            r"$H v=\lambda Mv$: changing M changes generalized curvature.",
            f"Exact-line-search q* bound = {summary['exact_line_search_factor']:.3g}; "
            "not an Armijo guarantee.",
            result.message,
        ])
        foreground = "#e1e9f2" if self.theme == "dark" else "#263547"
        ax.text(0.025, 0.97, "\n".join(lines), va="top", ha="left",
                transform=ax.transAxes, color=foreground, fontsize=6.6,
                linespacing=1.15, family="monospace", wrap=True)

    def create_plot_gallery(self) -> dict[str, Figure]:
        """Erzeuge die Diagramme des Vergleichs als einzelne Figuren."""
        objective = self.objective
        figures: dict[str, Figure] = {}

        contour_figure, contour_ax = plt.subplots(figsize=(9, 7))
        extent = self.plot_range
        x_star = objective.minimizer
        x_values = np.linspace(x_star[0] - extent, x_star[0] + extent, 180)
        y_values = np.linspace(x_star[1] - extent, x_star[1] + extent, 180)
        xx, yy = np.meshgrid(x_values, y_values)
        values = (
            0.5 * (
                objective.H[0, 0] * xx**2
                + 2 * objective.H[0, 1] * xx * yy
                + objective.H[1, 1] * yy**2
            )
            + objective.b[0] * xx
            + objective.b[1] * yy
            + objective.c
        )
        if self.visible["levels"]:
            positive_levels = np.geomspace(
                max((float(values.max()) - objective.minimum) * 1e-5, 1e-9),
                max(float(values.max()) - objective.minimum, 1e-8),
                32,
            )
            contour_ax.contour(
                xx, yy, values, levels=objective.minimum + positive_levels,
                colors="#718096", linewidths=0.8, alpha=0.8,
            )
        for index, (name, result) in enumerate(zip(METRIC_NAMES, self.results)):
            points = result.points
            if self.visible["trajectory"]:
                contour_ax.plot(
                    points[:, 0], points[:, 1], "-o", color=COLORS[index],
                    linewidth=1.6, markersize=3.5, label=name,
                )
            selected = points[min(self.iteration_index, len(points) - 1)]
            contour_ax.scatter(
                *selected, s=65, color=COLORS[index], edgecolor="white",
                linewidth=0.8, zorder=5,
            )
        selected_result = self.results[self.metric_index]
        selected_iteration = selected_result.iterations[
            min(self.iteration_index, len(selected_result.iterations) - 1)
        ]
        vector_scale = extent / max(
            np.linalg.norm(selected_iteration.gradient),
            np.linalg.norm(selected_iteration.direction),
            1.0,
        )
        if self.visible["gradient"] and np.linalg.norm(selected_iteration.gradient) > 0:
            contour_ax.quiver(
                *selected_iteration.x,
                *(selected_iteration.gradient * vector_scale),
                color="#d62780", angles="xy", scale_units="xy", scale=1,
                width=0.007, label="Gradient",
            )
        if self.visible["direction"] and np.linalg.norm(selected_iteration.direction) > 0:
            contour_ax.quiver(
                *selected_iteration.x,
                *(selected_iteration.direction * vector_scale * 2.5),
                color="#d4a900", angles="xy", scale_units="xy", scale=1,
                width=0.007, label="Abstiegsrichtung",
            )
        if self.visible["minimizer"]:
            contour_ax.scatter(
                *x_star, marker="*", s=180, color="#ef6c8e",
                edgecolor="white", linewidth=0.8, label="Minimierer",
                zorder=6,
            )
        contour_ax.set_aspect("equal", adjustable="box")
        contour_ax.set_xlim(x_star[0] - extent, x_star[0] + extent)
        contour_ax.set_ylim(x_star[1] - extent, x_star[1] + extent)
        contour_ax.set_title("Höhenlinien und Iterationswege")
        contour_ax.set_xlabel("$x_1$")
        contour_ax.set_ylabel("$x_2$")
        contour_ax.grid(alpha=0.16)
        contour_ax.legend()
        contour_figure.tight_layout()
        figures["contours"] = contour_figure

        surface_figure = plt.figure(figsize=(10, 7))
        surface_ax = surface_figure.add_subplot(111, projection="3d")
        surface_x = np.linspace(x_star[0] - extent, x_star[0] + extent, 70)
        surface_y = np.linspace(x_star[1] - extent, x_star[1] + extent, 70)
        surface_xx, surface_yy = np.meshgrid(surface_x, surface_y)
        surface_values = (
            0.5 * (
                objective.H[0, 0] * surface_xx**2
                + 2 * objective.H[0, 1] * surface_xx * surface_yy
                + objective.H[1, 1] * surface_yy**2
            )
            + objective.b[0] * surface_xx
            + objective.b[1] * surface_yy
            + objective.c
        )
        surface_ax.plot_surface(
            surface_xx, surface_yy, surface_values,
            cmap="viridis", alpha=0.5, linewidth=0, antialiased=True,
        )
        for index, (name, result) in enumerate(zip(METRIC_NAMES, self.results)):
            points = result.points
            path_values = np.asarray([objective.value(point) for point in points])
            if self.visible["trajectory"]:
                surface_ax.plot(
                    points[:, 0], points[:, 1], path_values,
                    color=COLORS[index], marker="o", markersize=3,
                    linewidth=1.8, label=name,
                )
        if self.visible["minimizer"]:
            surface_ax.scatter(
                *x_star, objective.minimum, marker="*", s=110,
                color="#ef6c8e", edgecolor="white", label="Minimierer",
            )
        surface_ax.set_title("Zielfläche und Iterationswege")
        surface_ax.set_xlabel("$x_1$")
        surface_ax.set_ylabel("$x_2$")
        surface_ax.set_zlabel("$f(x)$")
        surface_ax.view_init(elev=self._surface_view[0], azim=self._surface_view[1])
        surface_ax.legend()
        surface_figure.tight_layout()
        figures["surface"] = surface_figure

        armijo_figure, armijo_ax = plt.subplots(figsize=(9, 6))
        search = selected_iteration.armijo
        if search is None:
            armijo_ax.text(
                0.5, 0.5, "Am Minimierer ist keine Schrittweitensuche nötig.",
                ha="center", va="center", transform=armijo_ax.transAxes,
            )
        else:
            upper = float(search.trials[0])
            lower = float(search.trials[-1] * self.beta)
            trial_steps = np.geomspace(max(lower, 1e-12), upper, 240)
            trial_values = np.asarray([
                objective.value(
                    selected_iteration.x + step * selected_iteration.direction
                )
                for step in trial_steps
            ])
            sufficient_decrease = (
                selected_iteration.value
                + self.sigma * trial_steps
                * selected_iteration.directional_derivative
            )
            armijo_ax.plot(
                trial_steps, trial_values, color="#2878b5", linewidth=2,
                label="$f(x_k+t d_k)$",
            )
            armijo_ax.plot(
                trial_steps, sufficient_decrease, "--", color="#e28b20",
                linewidth=1.8, label="$f(x_k)+\\sigma t\\nabla f^T d_k$",
            )
            for step, trial_value, bound in zip(
                search.trials, search.values, search.bounds
            ):
                accepted = trial_value <= bound
                armijo_ax.scatter(
                    step, trial_value, marker="o" if accepted else "x",
                    color="#31904c" if accepted else "#d12b3f", s=60, zorder=5,
                )
            if search.accepted:
                armijo_ax.axvline(
                    search.step, color="#31904c", alpha=0.65, linestyle=":",
                )
                armijo_ax.annotate(
                    f"Akzeptiert: t={search.step:.3g}",
                    (search.step, search.values[-1]), xytext=(8, 10),
                    textcoords="offset points",
                )
            armijo_ax.set_xscale("log")
            armijo_ax.set_xlabel("Schrittweite $t$ (logarithmisch)")
            armijo_ax.set_ylabel("Funktionswert / Armijo-Schranke")
            armijo_ax.legend()
        armijo_ax.set_title(
            f"Armijo-Suche — {METRIC_NAMES[self.metric_index]}, "
            f"Iteration {min(self.iteration_index, len(selected_result.iterations) - 1)}"
        )
        armijo_ax.grid(alpha=0.2)
        armijo_figure.tight_layout()
        figures["armijo"] = armijo_figure

        spectrum_figure, spectrum_ax = plt.subplots(figsize=(9, 5))
        for index, (name, summary) in enumerate(zip(METRIC_NAMES, self.summaries)):
            minimum, maximum = summary["lambda_min"], summary["lambda_max"]
            spectrum_ax.plot(
                [minimum, maximum], [index, index], color=COLORS[index],
                linewidth=4, alpha=0.75,
            )
            spectrum_ax.scatter(
                [minimum, maximum], [index, index], color=COLORS[index], s=45,
            )
            spectrum_ax.annotate(
                f"{minimum:.3g}", (minimum, index), xytext=(0, -12),
                textcoords="offset points", ha="center",
            )
            spectrum_ax.annotate(
                f"{maximum:.3g}", (maximum, index), xytext=(0, 7),
                textcoords="offset points", ha="center",
            )
        spectrum_ax.set_xscale("log")
        spectrum_ax.set_yticks(range(len(METRIC_NAMES)), METRIC_NAMES)
        spectrum_ax.set_xlabel("Verallgemeinerte Eigenwerte $\\lambda$")
        spectrum_ax.set_title("Krümmungsvergleich: $Hv=\\lambda Mv$")
        spectrum_ax.grid(axis="x", alpha=0.25)
        spectrum_figure.tight_layout()
        figures["spectrum"] = spectrum_figure

        summary_figure, summary_ax = plt.subplots(figsize=(12, 3.5))
        summary_ax.axis("off")
        summary_columns = [
            "Metrik", "$\\kappa$", "$\\lambda_{min}$", "$\\lambda_{max}$",
            "Schritte", "$f(x_{final})$", "$f(x_{final})-f^*$",
        ]
        summary_rows = [
            [
                name,
                f"{summary['condition']:.3g}",
                f"{summary['lambda_min']:.3g}",
                f"{summary['lambda_max']:.3g}",
                str(int(summary["iterations"])),
                f"{summary['final_value']:.4g}",
                f"{summary['final_value'] - objective.minimum:.3g}",
            ]
            for name, summary in zip(METRIC_NAMES, self.summaries)
        ]
        table = summary_ax.table(
            cellText=summary_rows, colLabels=summary_columns,
            cellLoc="center", colLoc="center", loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.8)
        summary_ax.set_title("Vergleich der drei Metriken", pad=18)
        summary_figure.text(
            0.5, 0.06,
            "Die q*-Schranke gilt für die exakte Liniensuche, nicht als Armijo-Garantie.",
            ha="center", fontsize=9,
        )
        summary_figure.tight_layout(rect=(0, 0.13, 1, 1))
        figures["summary"] = summary_figure
        return figures

    def _apply_textboxes(self, _event=None) -> None:
        try:
            values = [float(box.text) for box in self._textbox_widgets]
            if not np.all(np.isfinite(values)):
                raise ValueError("All values must be finite.")
            (h11, h12, h22, b1, b2, constant,
             m11, m12, m22, x01, x02, plot_range,
             alpha_0, beta, sigma, max_backtracks, max_iterations, speed) = values
            if max_backtracks < 0 or max_backtracks != int(max_backtracks):
                raise ValueError("Backtracks must be a nonnegative integer.")
            if max_iterations < 0 or max_iterations != int(max_iterations):
                raise ValueError("Iterations must be a nonnegative integer.")
            if speed < 1 or speed != int(speed) or plot_range <= 0:
                raise ValueError("Plot range must be positive; animation speed must be a positive integer.")
            if not 0 < beta < 1 or not 0 < sigma < 1 or alpha_0 <= 0:
                raise ValueError("Use alpha0 > 0 and 0 < beta, sigma < 1.")
            proposed_H = np.array([[h11, h12], [h12, h22]])
            proposed_M = np.array([[m11, m12], [m12, m22]])
            proposed_objective = Quadratic(proposed_H, np.array([b1, b2]), constant)
            validate_metric(proposed_M, 2)
            np.linalg.cholesky(np.diag(np.diag(proposed_H)))
            self.H, self.b, self.c = proposed_H, np.array([b1, b2]), constant
            self.M, self.x0 = proposed_M, np.array([x01, x02])
            self.alpha_0, self.beta, self.sigma = alpha_0, beta, sigma
            self.max_backtracks, self.max_iterations = int(max_backtracks), int(max_iterations)
            self.plot_range, self.interval_ms = plot_range, int(speed)
            self.objective = proposed_objective
            self._timer.interval = self.interval_ms
            self._recompute()
            self._reset_contour_view = True
            self.fig._suptitle.set_text(
                "VARIABLES & CONTROLS  /  INNER PRODUCT M · ARMIJO · METRICS"
            )
            self._iteration_slider.valmax = max(
                1, len(self.results[self.metric_index].iterations) - 1
            )
            self._iteration_slider.ax.set_xlim(0, self._iteration_slider.valmax)
            self._iteration_slider.set_val(self.iteration_index)
            self._render()
        except (ValueError, np.linalg.LinAlgError) as error:
            self.fig.suptitle(
                f"Invalid settings — {error}", color="#ff8a80", fontsize=12
            )
            self.fig.canvas.draw_idle()

    def _select_metric(self, label: str) -> None:
        self.metric_index = METRIC_NAMES.index(label)
        self.iteration_index = min(
            self.iteration_index, len(self.results[self.metric_index].iterations) - 1
        )
        self._update_slider()
        self._render()

    def _select_iteration(self, value: float) -> None:
        self.iteration_index = min(
            int(value), len(self.results[self.metric_index].iterations) - 1
        )
        self._render()

    def _update_slider(self) -> None:
        maximum = max(1, len(self.results[self.metric_index].iterations) - 1)
        self._iteration_slider.valmax = maximum
        self._iteration_slider.ax.set_xlim(0, maximum)
        self._iteration_slider.set_val(min(self.iteration_index, maximum))

    def _step_back(self, _event=None) -> None:
        self.iteration_index = max(0, self.iteration_index - 1)
        self._update_slider()
        self._render()

    def _step_forward(self, _event=None) -> None:
        maximum = len(self.results[self.metric_index].iterations) - 1
        self.iteration_index = min(maximum, self.iteration_index + 1)
        self._update_slider()
        self._render()

    def _advance_animation(self) -> bool:
        maximum = len(self.results[self.metric_index].iterations) - 1
        if self.iteration_index >= maximum:
            self._timer.stop()
            return False
        self.iteration_index += 1
        self._update_slider()
        self._render()
        return True

    def _toggle_play(self, _event=None) -> None:
        if self._timer is not None and self._timer.running:
            self._timer.stop()
        else:
            if self.iteration_index >= len(self.results[self.metric_index].iterations) - 1:
                self.iteration_index = 0
                self._update_slider()
            self._timer.start()

    def _reset(self, _event=None) -> None:
        self.iteration_index = 0
        self._update_slider()
        self._render()

    def _toggle_theme(self, _event=None) -> None:
        self.theme = "light" if self.theme == "dark" else "dark"
        self._render()

    def _toggle_visibility(self, label: str) -> None:
        key = self._visibility_keys[label]
        self.visible[key] = not self.visible[key]
        self._render()

    def _select_path_point(
        self, metric_index: int, indices: NDArray, *, from_contour: bool = False
    ) -> None:
        self.metric_index = metric_index
        self.iteration_index = min(
            int(indices[0]), len(self.results[metric_index].iterations) - 1
        )
        self._path_picked = from_contour
        self._metric_radio.set_active(metric_index)
        self._update_slider()
        self._render()

    def _on_pick(self, event) -> None:
        selection = self._picked_paths.get(id(event.artist))
        if selection is None or not len(event.ind):
            if event.artist is self._initial_artist:
                self._dragging = True
            return
        self._select_path_point(
            selection[0], event.ind, from_contour=event.artist.axes is self.ax_contour
        )

    def _on_press(self, event) -> None:
        if event.inaxes is self.ax_contour and event.button == 1:
            toolbar = getattr(event.canvas, "toolbar", None)
            if toolbar is not None and toolbar.mode:
                return
            if self._path_picked:
                self._path_picked = False
                return
            for artist in self.ax_contour.lines:
                selection = self._picked_paths.get(id(artist))
                if selection is not None:
                    contains, details = artist.contains(event)
                    if contains and len(details["ind"]):
                        self._select_path_point(selection[0], details["ind"], from_contour=True)
                        return
            if self._initial_artist is not None and self._initial_artist.contains(event)[0]:
                self._dragging = True
            else:
                self._set_initial_point(event.xdata, event.ydata)

    def _on_motion(self, event) -> None:
        if self._dragging and event.inaxes is self.ax_contour:
            self._set_initial_point(event.xdata, event.ydata)

    def _on_release(self, _event) -> None:
        self._dragging = False

    def _set_initial_point(self, x: float | None, y: float | None) -> None:
        if x is None or y is None:
            return
        self.x0 = np.array([x, y])
        self._reset_contour_view = False
        self._recompute()
        self._update_textbox_values()
        self._update_slider()
        self._render()

    def _update_textbox_values(self) -> None:
        values = (
            self.H[0, 0], self.H[0, 1], self.H[1, 1], self.b[0], self.b[1], self.c,
            self.M[0, 0], self.M[0, 1], self.M[1, 1], self.x0[0], self.x0[1],
            self.plot_range, self.alpha_0, self.beta, self.sigma, self.max_backtracks,
            self.max_iterations, self.interval_ms,
        )
        for box, value in zip(self._textbox_widgets, values):
            box.set_val(f"{value:g}")

    def _on_key(self, event) -> None:
        if event.key in ("right", "n"):
            self._step_forward()
        elif event.key in ("left", "p"):
            self._step_back()
        elif event.key == " ":
            self._toggle_play()

    def show(self) -> Figure:
        """Zeige Höhenlinien, 3D-Fläche sowie Variablen- und Steuerungsansicht."""
        plt.show()
        return self.fig


def create_laboratory(**kwargs) -> QuadraticLaboratory:
    """Erzeuge ein quadratisches Optimierungsbeispiel."""
    return QuadraticLaboratory(**kwargs)
