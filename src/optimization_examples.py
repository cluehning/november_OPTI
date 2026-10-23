"""Zielfunktionen und Diagramme zu den Gradientenverfahren."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def quadratic_function(x, derivatives, Q, c, gamma):
    """Werte 1/2 x^T Q x + c^T x + gamma samt angeforderten Ableitungen aus."""
    values = {}
    if derivatives[0]:
        values["function"] = 0.5 * x.dot(Q.dot(x)) + c.dot(x) + gamma
    if derivatives[1]:
        values["derivative"] = Q.dot(x) + c
    if derivatives[2]:
        values["Hessian"] = Q
    return values


def himmelblau(x, derivatives):
    """Himmelblau-Funktion mit Gradient und Hesse-Matrix."""
    values = {}
    x1, x2 = x
    first_term = x1**2 + x2 - 11
    second_term = x1 + x2**2 - 7
    if derivatives[0]:
        values["function"] = first_term**2 + second_term**2
    if derivatives[1]:
        # Die Kettenregel wendet die äußere Ableitung der Quadrate auf beide
        # inneren Polynome an. Der Gradient zeigt lokal in die Richtung des
        # stärksten Anstiegs; sein Negatives ist die Abstiegsrichtung.
        values["derivative"] = np.array(
            [
                4 * x1 * first_term + 2 * second_term,
                2 * first_term + 4 * x2 * second_term,
            ]
        )
    if derivatives[2]:
        # Die zweiten Ableitungen beschreiben, wie sich der Gradient ändert.
        # Sie helfen beim Vergleich von Metriken, sind bei einem beliebigen
        # Startpunkt aber nicht automatisch ein guter Vorkonditionierer.
        values["Hessian"] = np.array(
            [
                [12 * x1**2 + 4 * (x2 - 11) + 2, 4 * (x1 + x2)],
                [4 * (x1 + x2), 12 * x2**2 + 4 * (x1 - 7) + 2],
            ]
        )
    return values


def rosenbrock(x, derivatives, parameters=None):
    """Rosenbrock-Funktion; ``a`` und ``b`` steuern ihre Form."""
    coefficients = {} if parameters is None else parameters
    a = coefficients.get("a", 1)
    b = coefficients.get("b", 100)
    values = {}
    x1, x2 = x
    valley_error = x2 - x1**2
    if derivatives[0]:
        values["function"] = (a - x1) ** 2 + b * valley_error**2
    if derivatives[1]:
        values["derivative"] = np.array(
            [
                -2 * (a - x1) - 4 * b * x1 * valley_error,
                2 * b * valley_error,
            ]
        )
    if derivatives[2]:
        values["Hessian"] = np.array(
            [
                [2 - 4 * b * (x2 - 3 * x1**2), -4 * b * x1],
                [-4 * b * x1, 2 * b],
            ]
        )
    return values


def generalized_condition_number(Q, M):
    """Berechne die Konditionszahl des verallgemeinerten Problems Qv=lambda Mv."""
    hessian = np.asarray(Q, dtype=float)
    metric = np.asarray(M, dtype=float)
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError("Q muss quadratisch sein.")
    if metric.shape != hessian.shape or not np.allclose(metric, metric.T):
        raise ValueError("M muss symmetrisch sein und dieselbe Größe wie Q haben.")
    factor = np.linalg.cholesky(metric)
    # Mit M = L L^T wird Qv=lambda Mv auf
    # L^{-1} Q L^{-T} y=lambda y transformiert. Dessen gewöhnliche
    # Eigenwerte sind die verallgemeinerten Eigenwerte des Problems.
    left_whitened = np.linalg.solve(factor, hessian)
    whitened = np.linalg.solve(factor, left_whitened.T).T
    eigenvalues = np.linalg.eigvalsh((whitened + whitened.T) / 2)
    if eigenvalues[0] <= 0:
        raise ValueError("Q muss für diese Konditionszahl positiv definit sein.")
    return float(eigenvalues[-1] / eigenvalues[0])


def plot_2d_iterates_contours(f, histories, labels):
    """Zeige zweidimensionale Iterierte zusammen mit Funktionshöhenlinien."""
    all_iterates = np.vstack(
        [np.vstack(history["iterates"]) for history in histories]
    )
    max_abs_component = float(np.abs(all_iterates).max())
    if max_abs_component == 0:
        max_abs_component = 1.0
    grid = np.linspace(-1.5 * max_abs_component, 1.5 * max_abs_component, 251)
    X, Y = np.meshgrid(grid, grid)
    Z = np.empty((grid.size, grid.size))
    for i, x_component in enumerate(grid):
        for j, y_component in enumerate(grid):
            Z[j, i] = f(
                np.array([x_component, y_component]),
                [True, False, False],
            )["function"]

    plt.figure()
    for history in histories:
        # Die Konturwerte stammen aus den besuchten Punkten: Dadurch zeigen
        # die Linien direkt, wie sich die Funktionshöhe entlang des Laufs ändert.
        contour_values = sorted(set(history["objective_values"][:: max(
            1, len(history["objective_values"]) // 20
        )]))
        if contour_values:
            plt.contour(X, Y, Z, levels=contour_values)
        points = np.vstack(history["iterates"])
        plt.plot(points[:, 0], points[:, 1], "s-")

    plt.gca().set_aspect("equal", "box")
    plt.title("Iterierte und Höhenlinien der Zielfunktion")
    plt.legend(labels)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")


def plot_f_val_diffs(histories, reference_values, labels, condition_numbers=None):
    """Vergleiche Funktionswertfehler und optional die quadratische Schranke."""
    plt.figure()
    for history, reference_value in zip(histories, reference_values):
        differences = np.asarray(history["objective_values"]) - reference_value
        # Die Differenz ist bei einer quadratischen Funktion mit exaktem Minimum
        # gleich der halben quadrierten Energienorm des Fehlers.
        plt.semilogy(differences)

    bound_labels = []
    if condition_numbers is not None:
        plt.gca().set_prop_cycle(None)
        for history, reference_value, condition_number, label in zip(
            histories, reference_values, condition_numbers, labels
        ):
            factor = (condition_number - 1) / (condition_number + 1)
            initial_error = history["objective_values"][0] - reference_value
            bound = initial_error * factor ** (
                2 * np.arange(len(history["objective_values"]))
            )
            plt.semilogy(bound, "--")
            bound_labels.append(label + " (theoretische Schranke)")

    plt.title("Differenz der Funktionswerte")
    plt.xlabel("Iteration")
    plt.ylabel("$f(x_k)-f_{min}$")
    plt.legend(list(labels) + bound_labels)


def plot_step_sizes(histories, labels):
    """Stelle die akzeptierten Schrittweiten jeder Iteration dar."""
    plt.figure()
    for history in histories:
        plt.plot(history["step_lengths"])
    plt.title("Akzeptierte Schrittweiten")
    plt.xlabel("Iteration")
    plt.ylabel("Schrittweite")
    plt.legend(labels)


def plot_grad_norms(histories, labels):
    """Stelle die M-Norm des vorkonditionierten Gradienten logarithmisch dar."""
    plt.figure()
    for history in histories:
        plt.semilogy(history["gradient_norms"])
    plt.title("Norm des vorkonditionierten Gradienten")
    plt.xlabel("Iteration")
    plt.ylabel("$\\|M^{-1}\\nabla f(x_k)\\|_M$")
    plt.legend(labels)
