#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Rana(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Rana}}(x) = \sum_{i=1}^{n} \left[x_{i} \sin\left(\sqrt{\lvert{x_{1} - x_{i} + 1}\rvert}\right)
        \cos\left(\sqrt{\lvert{x_{1} + x_{i} + 1}\rvert}\right) + \left(x_{1} + 1\right) \sin\left(\sqrt{\lvert{x_{1} + x_{i} +
        1}\rvert}\right) \cos\left(\sqrt{\lvert{x_{1} - x_{i} +1}\rvert}\right)\right]

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-500.0, 500.0]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x_i) = -928.5478` for :math:`x = [-300.3376, 500]`.
    """
    name = "Qing Function"
    latex_formula = r'f_{\text{Rana}}(x) = '
    latex_formula_dimension = r'd = n'
    latex_formula_bounds = r'x_i \in [-10, 10, ..., 10]'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 1.0'
    continuous = True
    linear = False
    convex = True
    unimodal = False
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True  # Number of ambiguous peaks, unknown # peaks

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
        self.f_global = -500.8021602966615
        self.x_global = np.array([-300.3376, 500.])

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        t1 = np.sqrt(np.abs(x[1:] + x[: -1] + 1))
        t2 = np.sqrt(np.abs(x[1:] - x[: -1] + 1))
        v = (x[1:] + 1) * np.cos(t2) * np.sin(t1) + x[:-1] * np.cos(t1) * np.sin(t2)
        return np.sum(v)


class Rosenbrock(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Rosenbrock}}(\mathbf{x}) = \sum_{i=1}^{n-1} \left(100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2\right)

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(\mathbf{x}) = 0` for :math:`x_i = 1` for :math:`i = 1, ..., n`
    """

    name = "Rosenbrock Function"
    latex_formula = r'f_{\text{Rosenbrock}}(\mathbf{x}) = \sum_{i=1}^{n-1} \left(100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2\right)'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(1, 1, ...,1) = 0'
    continuous = True
    linear = False
    convex = False
    unimodal = True
    separable = False

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = False

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-10., 10.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.ones(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2)


class Rastrigin(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Rastrigin}}(\mathbf{x}) = \sum_{i=1}^n \left(x_i^2 - 10\cos(2\pi x_i) + 10\right)

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-5.12, 5.12]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(\mathbf{x}) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """

    name = "Rastrigin Function"
    latex_formula = r'f_{\text{Rastrigin}}(\mathbf{x}) = \sum_{i=1}^n \left(x_i^2 - 10\cos(2\pi x_i) + 10\right)'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-5.12, 5.12], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = True

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5.12, 5.12] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0)
