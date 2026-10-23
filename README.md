# Gradient Descent: An Interactive Optimization Lab

This project explores how gradient descent behaves when the geometry of a problem changes. Its browser interface compares several metrics and lets you switch between a quadratic function, Himmelblau's function, and the Rosenbrock function.

The main idea is to make the optimization process visible: instead of seeing only a final answer, I can follow each iteration, inspect the Armijo step-size search, and compare how different metrics affect the path.

## Why compare metrics?

For a differentiable function \(f\), ordinary steepest descent follows the negative gradient. A symmetric positive-definite matrix \(M\) defines a different metric and changes the search direction to

$$
d_k=-M^{-1}\nabla f(x_k).
$$

The direction is still a descent direction, but its components are scaled by the chosen metric. Comparing these paths helps show how a suitable metric can improve the behavior of gradient descent.

## The objective functions

The browser lets me choose between three two-dimensional examples:

* **Quadratic:** \(f(x)=\tfrac12 x^\mathsf{T}Hx+b^\mathsf{T}x+c\), with editable coefficients. The matrix \(H\) must be positive definite.
* **Himmelblau:** \(f(x)=(x_1^2+x_2-11)^2+(x_1+x_2^2-7)^2\).
* **Rosenbrock:** \(f(x)=(a-x_1)^2+b(x_2-x_1^2)^2\), with editable positive parameters \(a\) and \(b\).

For each function, the interface compares Euclidean distance, a user-defined metric \(M\), and a diagonal metric. For the quadratic example, the diagonal metric is the Jacobi preconditioner \(\operatorname{diag}(H)\). For Himmelblau and Rosenbrock, it is formed from the absolute Hessian diagonal at the starting point.

## What I can explore

1. **Choose a function and starting point.** The interface supplies a starting point for each function, which can then be changed.
2. **Compare the paths.** Contour lines, the three-dimensional surface, and coordinate plots show how each metric moves toward a low function value.
3. **Inspect Armijo backtracking.** The step-size plot shows the objective along the current search direction, the Armijo bound, and the trial steps.
4. **Review the results.** The summary reports iteration counts and final values. For non-quadratic functions it also reports the final gradient norm and whether each run converged.

Himmelblau and Rosenbrock can have multiple local minima. The marked point is the best final result among the three runs; it is not guaranteed to be a global minimum. Generalized eigenvalues and condition numbers are shown only for the quadratic function.

## Quick start

Open `Interactive_Optimization_Lab.html` in a browser, or open `Gradientenverfahren.ipynb` and run its final cell to display the browser interface in the notebook. The plots use Plotly from a CDN, so an internet connection is needed.

The Python modules and tests can be used separately:

```powershell
python -m pip install -r requirements-lab.txt
python -m unittest discover -s tests -v
```

## Project files

```text
OPTI/
├── Interactive_Optimization_Lab.html  # Browser interface and JavaScript calculations
├── Gradientenverfahren.ipynb          # Algorithm explanations and embedded browser
├── OPTIMIZATION_LAB.md                # Detailed German project documentation
├── requirements-lab.txt               # Python dependencies
├── src/
│   ├── gradient_methods.py            # Python gradient and step-size methods
│   ├── optimization_examples.py       # Objective functions and plotting helpers
│   └── optimization_lab.py            # Separate Matplotlib optimization laboratory
└── tests/
    └── test_optimization_lab.py       # Tests for the Python optimization laboratory
```

## How the browser and Python code relate

The browser page calculates the objective functions and optimization steps in JavaScript; it does not import or run the Python modules. The Python implementations are useful for studying and testing the algorithms separately. Their corresponding browser calculations are maintained in the HTML file, so the two implementations should be checked against each other if either changes.

For setup details and the German-language explanation, see [OPTIMIZATION_LAB.md](OPTIMIZATION_LAB.md).
