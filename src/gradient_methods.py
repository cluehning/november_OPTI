"""Vorkonditionierte Gradientenverfahren mit zwei Schrittweitenstrategien."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np

FloatArray = np.ndarray


class FunctionValues(TypedDict, total=False):
    function: float
    derivative: FloatArray | float
    Hessian: FloatArray | float


Objective = Callable[[FloatArray, list[bool]], FunctionValues]
StepLengthRule = Callable[[Callable, dict[str, float]], tuple[float, int]]


def _float_parameter(
    options: dict[str, object], name: str, default: float
) -> float:
    """Lies einen numerischen Parameter mit einer verständlichen Fehlermeldung."""
    value = options.get(name, default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        result = float(value)
        if np.isfinite(result):
            return result
    raise ValueError(f"{name} muss eine endliche Zahl sein.")


def _required_scalar(values: FunctionValues, name: str) -> float:
    """Lies einen von der Zielfunktion angeforderten skalaren Wert aus."""
    value = values.get(name)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    raise ValueError(f"Die Zielfunktion muss den skalaren Wert {name!r} liefern.")


def gradient_descent(
    f: Objective,
    x0: FloatArray,
    step_length_rule: StepLengthRule,
    preconditioner: FloatArray,
    parameters: dict[str, object] | None = None,
) -> dict[str, object]:
    """Minimiere f mit dem vorkonditionierten Gradientenverfahren.

    ``preconditioner`` ist die symmetrisch positiv definite Matrix M. Die
    Suchrichtung ist ``-M^{-1} grad f``; M verändert also die Geometrie, in
    der die lokale Steigung in eine Abstiegsrichtung übersetzt wird.
    """
    options = {} if parameters is None else dict(parameters)
    x = np.asarray(x0, dtype=float).copy()
    metric = np.asarray(preconditioner, dtype=float)
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 muss ein nichtleerer endlicher Vektor sein.")
    if metric.shape != (x.size, x.size) or not np.all(np.isfinite(metric)):
        raise ValueError("Der Vorkonditionierer muss eine endliche Matrix passender Größe sein.")
    if not np.allclose(metric, metric.T, rtol=1e-10, atol=1e-12):
        raise ValueError("Der Vorkonditionierer muss symmetrisch sein.")
    metric = (metric + metric.T) / 2
    try:
        np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as error:
        raise ValueError("Der Vorkonditionierer muss positiv definit sein.") from error

    atol_x = _float_parameter(options, "atol_x", 1e-6)
    rtol_x = _float_parameter(options, "rtol_x", 1e-6)
    atol_f = _float_parameter(options, "atol_f", 1e-6)
    rtol_f = _float_parameter(options, "rtol_f", rtol_x**2)
    atol_gradf = _float_parameter(options, "atol_gradf", 1e-6)
    rtol_gradf = _float_parameter(options, "rtol_gradf", 1e-6)
    if min(atol_x, rtol_x, atol_f, rtol_f, atol_gradf, rtol_gradf) < 0:
        raise ValueError("Abbruchtoleranzen müssen nichtnegativ sein.")
    max_iterations = options.get("max_iterations", 1000)
    verbosity = options.get("verbosity", "quiet")
    keep_history = bool(options.get("keep_history", False))

    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 0:
        raise ValueError("max_iterations muss eine nichtnegative ganze Zahl sein.")
    if verbosity not in ("quiet", "verbose"):
        raise ValueError("verbosity muss 'quiet' oder 'verbose' sein.")

    history: dict[str, list] = {
        "iterates": [],
        "objective_values": [],
        "gradient_norms": [],
        "step_lengths": [],
    }
    x_previous: FloatArray | None = None
    f_previous: float | None = None
    initial_gradient_norm: float | None = None
    iteration = 0
    exitflag: int | None = None
    function_value = float("nan")
    derivative = np.zeros_like(x)
    gradient = np.zeros_like(x)
    norm_gradient = float("nan")

    if verbosity == "verbose":
        print(
            " ITER          OBJ    NORM_GRAD    NORM_CORR "
            "   OBJ_CHNG"
        )

    while exitflag is None:
        # f liefert Wert und Gradient gemeinsam: So verwenden sowohl die
        # Abbruchprüfung als auch die Richtung dieselben lokalen Daten.
        values = f(x, [True, True, False])
        function_value = _required_scalar(values, "function")
        raw_derivative = values.get("derivative")
        if raw_derivative is None:
            raise ValueError("Die Zielfunktion muss den Gradienten liefern.")
        derivative = np.asarray(raw_derivative, dtype=float)
        if derivative.shape != x.shape or not np.all(np.isfinite(derivative)):
            raise ValueError("Der Gradient muss ein endlicher Vektor wie x sein.")
        if not np.isfinite(function_value):
            raise ValueError("Die Zielfunktion muss an jedem Iterationspunkt endlich sein.")

        # M^{-1} grad f ist der Gradient in der durch M festgelegten Geometrie.
        # Seine M-Norm ist sqrt(grad(f)^T M^{-1} grad(f)); sie misst daher
        # dieselbe Steigung, die auch die Suchrichtung bestimmt.
        gradient = np.linalg.solve(metric, derivative)
        norm_gradient_squared = float(derivative @ gradient)
        if norm_gradient_squared < -1e-12:
            raise ValueError("Der Vorkonditionierer muss positiv definit sein.")
        norm_gradient = float(np.sqrt(max(0.0, norm_gradient_squared)))
        if initial_gradient_norm is None:
            initial_gradient_norm = norm_gradient

        if keep_history:
            history["iterates"].append(x.copy())
            history["objective_values"].append(function_value)
            history["gradient_norms"].append(norm_gradient)
        if verbosity == "verbose":
            print(
                f"{iteration:4d}  {function_value:11.4e}  "
                f"{norm_gradient:11.4e}",
                end="",
            )

        # Die absolute Toleranz fängt kleine Gradienten nahe null ab. Der
        # relative Anteil skaliert die Prüfung mit der anfänglichen Steigung.
        if norm_gradient <= atol_gradf + rtol_gradf * initial_gradient_norm:
            exitflag = 1
        elif x_previous is not None and f_previous is not None:
            # Beide Änderungen müssen klein sein: Ein kleiner Funktionsabfall
            # allein kann auch durch eine sehr kleine Schrittweite entstehen.
            delta_x = x - x_previous
            norm_delta_x = float(np.sqrt(max(0.0, delta_x @ metric @ delta_x)))
            delta_f = f_previous - function_value
            if verbosity == "verbose":
                print(f"  {norm_delta_x:11.4e}  {-delta_f:11.4e}", end="")
            if (
                delta_f < atol_f + rtol_f * abs(f_previous)
                and norm_delta_x < atol_x + rtol_x * np.sqrt(
                    max(0.0, x_previous @ metric @ x_previous)
                )
            ):
                exitflag = 0
        if exitflag is not None:
            if verbosity == "verbose":
                print()
            break

        # iteration zählt bereits angenommene Schritte. Der aktuelle Punkt
        # wurde oben ausgewertet, daher bleibt das Ergebnis auch beim Erreichen
        # des Iterationslimits konsistent mit dem zurückgegebenen x.
        if iteration >= max_iterations:
            exitflag = 2
            if verbosity == "verbose":
                print("  Iterationslimit erreicht.")
            break

        direction = -gradient

        # Entlang x + t d wird aus f eine eindimensionale Funktion phi(t).
        # Die Kettenregel liefert phi'(0) = grad(f)^T d; bei dieser Richtung
        # ist das gleich -grad(f)^T M^{-1} grad(f) und damit negativ.
        def phi(t: float, derivatives: list[bool]) -> FunctionValues:
            line_values = f(x + t * direction, derivatives)
            if derivatives[1]:
                raw_derivative = line_values.get("derivative")
                if raw_derivative is None:
                    raise ValueError("Die Zielfunktion muss den Gradienten liefern.")
                line_values["derivative"] = float(
                    np.asarray(raw_derivative, dtype=float) @ direction
                )
            if derivatives[2]:
                raw_hessian = line_values.get("Hessian")
                if raw_hessian is None:
                    raise ValueError("Die Zielfunktion muss die Hesse-Matrix liefern.")
                hessian = np.asarray(raw_hessian, dtype=float)
                line_values["Hessian"] = float(direction @ hessian @ direction)
            return line_values

        # Diese Werte sind in der aktuellen Iteration bereits bekannt. Damit
        # vermeiden wir eine erneute Funktions- und Gradienten-Auswertung bei t=0.
        reusable_values = {
            "phi0": function_value,
            "dphi0": -norm_gradient_squared,
        }
        step, step_exitflag = step_length_rule(phi, reusable_values)
        if step_exitflag != 0:
            raise RuntimeError(
                f"Die Schrittweitensuche ist mit Status {step_exitflag} fehlgeschlagen."
            )
        if not np.isfinite(step) or step <= 0:
            raise RuntimeError("Die Schrittweitensuche muss eine positive endliche Weite liefern.")

        if keep_history:
            history["step_lengths"].append(float(step))
        x_previous = x.copy()
        f_previous = function_value
        x = x + step * direction
        iteration += 1
        if verbosity == "verbose":
            print(f"  Schrittweite t = {step:.4e}")

    result: dict[str, object] = {
        "solution": x,
        "function": function_value,
        "gradient": derivative,
        "norm_gradient": norm_gradient,
        "iter": iteration,
        "exitflag": exitflag,
    }
    if keep_history:
        result["history"] = history
    return result


def exact_step_length_quadratic(
    phi: Callable,
    reusables: dict[str, float] | None = None,
) -> tuple[float, int]:
    """Bestimme das Minimum einer streng konvexen quadratischen Schnittfunktion."""
    cached = {} if reusables is None else reusables

    # Für phi(t) = phi(0) + phi'(0)t + a t² folgt aus phi(1):
    # a = phi(1) - phi(0) - phi'(0). Der Minimierer -phi'(0)/(2a)
    # ist die exakte Schrittweite; anders als Backtracking gibt es hier
    # keine Folge von Versuchen.
    phi0 = cached.get("phi0")
    if phi0 is None:
        phi0 = _required_scalar(phi(0, [True, False, False]), "function")
    dphi0 = cached.get("dphi0")
    if dphi0 is None:
        dphi0 = _required_scalar(phi(0, [False, True, False]), "derivative")
    if dphi0 >= 0:
        raise ValueError("Die Schnittfunktion muss bei t=0 streng fallen.")
    phi1 = _required_scalar(phi(1, [True, False, False]), "function")
    curvature = phi1 - phi0 - dphi0
    if not np.isfinite(curvature) or curvature <= 0:
        raise ValueError("Die Schnittfunktion muss positive Krümmung besitzen.")
    step = -dphi0 / (2 * curvature)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Die exakte Schrittweite muss positiv und endlich sein.")
    return float(step), 0


def armijo_backtracking(
    phi: Callable,
    reusables: dict[str, float] | None = None,
    parameters: dict[str, object] | None = None,
) -> tuple[float, int]:
    """Suche per Backtracking die erste Schrittweite mit Armijo-Abstieg."""
    cached = {} if reusables is None else reusables
    options = {} if parameters is None else parameters
    sigma = _float_parameter(options, "sigma", 0.01)
    beta = _float_parameter(options, "beta", 0.5)
    initial_step = _float_parameter(options, "initial_step_length", 1.0)
    max_backtracks = options.get("max_iterations", 10000)
    verbosity = options.get("verbosity", "quiet")

    if not 0 < sigma < 1:
        raise ValueError("sigma muss strikt zwischen 0 und 1 liegen.")
    if not 0 < beta < 1:
        raise ValueError("beta muss strikt zwischen 0 und 1 liegen.")
    if not np.isfinite(initial_step) or initial_step <= 0:
        raise ValueError("Die anfängliche Schrittweite muss positiv und endlich sein.")
    if not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0:
        raise ValueError("max_iterations muss eine nichtnegative ganze Zahl sein.")
    if verbosity not in ("quiet", "verbose"):
        raise ValueError("verbosity muss 'quiet' oder 'verbose' sein.")

    phi0 = cached.get("phi0")
    if phi0 is None:
        phi0 = _required_scalar(phi(0, [True, False, False]), "function")
    dphi0 = cached.get("dphi0")
    if dphi0 is None:
        dphi0 = _required_scalar(phi(0, [False, True, False]), "derivative")
    if not np.isfinite(phi0) or not np.isfinite(dphi0) or dphi0 >= 0:
        raise ValueError("Die Schnittfunktion muss bei t=0 endlich und fallend sein.")

    # β verkleinert t geometrisch. So wird ein zunächst zu großer Schritt
    # schrittweise vorsichtiger, bis der tatsächliche Abstieg mindestens den
    # von σ geforderten Anteil des linearen Abstiegs erreicht.
    step = initial_step
    for iteration in range(max_backtracks + 1):
        phi_trial = _required_scalar(phi(step, [True, False, False]), "function")
        armijo_bound = phi0 + sigma * step * dphi0
        if verbosity == "verbose":
            print(
                f"Armijo-Versuch {iteration:4d}: "
                f"t={step:.4e}, phi(t)-phi(0)={phi_trial - phi0:.4e}"
            )
        if np.isfinite(phi_trial) and phi_trial <= armijo_bound:
            return float(step), 0
        if iteration == max_backtracks:
            return float(step), 1
        if phi_trial == phi0 and step / initial_step < 1e-12:
            return float(step), 2
        step *= beta
    raise RuntimeError("Unerreichbarer Zustand in der Armijo-Suche.")
