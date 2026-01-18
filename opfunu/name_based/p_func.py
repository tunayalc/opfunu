#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


def _u_penalty(x, a, k, m):
    x = np.asarray(x)
    return np.where(
        x > a,
        k * (x - a) ** m,
        np.where(x < -a, k * (-x - a) ** m, 0.0),
    )


class Parsopoulos(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Parsopoulos}}(x) = \cos(x_1)^2 + \sin(x_2)^2

    with :math:`x_i \in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: This function has infinite number of global minima in R2, at points
    :math:`\left(k\frac{\pi}{2}, \lambda \pi \right)`, where :math:`k = \pm1, \pm3, ...` and :math:`\lambda = 0, \pm1, \pm2, ...`

    In the given domain problem, function has 12 global minima all equal to zero.
    """
    name = "Parsopoulos Function"
    latex_formula = r'f_{\text{Parsopoulos}}(x) = \cos(x_1)^2 + \sin(x_2)^2'
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = False  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        return np.cos(x[0]) ** 2.0 + np.sin(x[1]) ** 2.0


class Penalized1(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        y_i = 1 + \frac{x_i + 1}{4}

        f(\mathbf{x}) = \frac{\pi}{n}\left[10\sin^2(\pi y_1) + \sum_{i=1}^{n-1}(y_i-1)^2\left(1 + 10\sin^2(\pi y_{i+1})\right) + (y_n - 1)^2\right] + \sum_{i=1}^n u(x_i, 10, 100, 4)
    """

    name = "Penalized 1 Function"
    latex_formula = r'f(\mathbf{x}) = \frac{\pi}{n}\left[10\sin^2(\pi y_1) + \sum_{i=1}^{n-1}(y_i-1)^2\left(1 + 10\sin^2(\pi y_{i+1})\right) + (y_n - 1)^2\right] + \sum_{i=1}^n u(x_i, 10, 100, 4)'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-50, 50], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(-1, -1, ..., -1) = 0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-50., 50.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = -np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        n = self.ndim
        y = 1.0 + (x + 1.0) / 4.0
        term1 = 10.0 * np.sin(np.pi * y[0]) ** 2
        term2 = np.sum((y[:-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * y[1:]) ** 2))
        term3 = (y[-1] - 1.0) ** 2
        penalty = np.sum(_u_penalty(x, a=10.0, k=100.0, m=4.0))
        return (np.pi / n) * (term1 + term2 + term3) + penalty


class Penalized2(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f(\mathbf{x}) = 0.1\left[\sin^2(3\pi x_1) + \sum_{i=1}^{n-1}(x_i-1)^2\left(1 + \sin^2(3\pi x_{i+1})\right) + (x_n - 1)^2\left(1 + \sin^2(2\pi x_n)\right)\right] + \sum_{i=1}^n u(x_i, 5, 100, 4)
    """

    name = "Penalized 2 Function"
    latex_formula = r'f(\mathbf{x}) = 0.1\left[\sin^2(3\pi x_1) + \sum_{i=1}^{n-1}(x_i-1)^2\left(1 + \sin^2(3\pi x_{i+1})\right) + (x_n - 1)^2\left(1 + \sin^2(2\pi x_n)\right)\right] + \sum_{i=1}^n u(x_i, 5, 100, 4)'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-50, 50], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1, 1, ..., 1) = 0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-50., 50.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        term1 = np.sin(3.0 * np.pi * x[0]) ** 2
        term2 = np.sum((x[:-1] - 1.0) ** 2 * (1.0 + np.sin(3.0 * np.pi * x[1:]) ** 2))
        term3 = (x[-1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * x[-1]) ** 2)
        penalty = np.sum(_u_penalty(x, a=5.0, k=100.0, m=4.0))
        return 0.1 * (term1 + term2 + term3) + penalty
