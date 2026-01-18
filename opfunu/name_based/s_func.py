#!/usr/bin/env python
# Created by "Thieu" at 17:31, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from opfunu.benchmark import Benchmark


class Salomon(Benchmark):
    """
    .. [1]  Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Salomon}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}


    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """
    name = "Qing Function"
    latex_formula = r'f_{\text{Salomon}}(x) = 1 - \cos \left (2 \pi \sqrt{\sum_{i=1}^{n} x_i^2} \right) + 0.1 \sqrt{\sum_{i=1}^n x_i^2}'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        u = np.sqrt(np.sum(x ** 2))
        return 1 - np.cos(2 * np.pi * u) + 0.1 * u


class Sphere(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Sphere}}(\mathbf{x}) = \sum_{i=1}^{n} x_i^2

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(\mathbf{x}) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """

    name = "Sphere Function"
    latex_formula = r'f_{\text{Sphere}}(\mathbf{x}) = \sum_{i=1}^{n} x_i^2'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = True

    differentiable = True
    scalable = True
    randomized_term = False
    parametric = False

    modality = False

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return np.sum(x ** 2)


class SumSquares(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{SumSquares}}(\mathbf{x}) = \sum_{i=1}^{n} i x_i^2

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(\mathbf{x}) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """

    name = "SumSquares Function"
    latex_formula = r'f_{\text{SumSquares}}(\mathbf{x}) = \sum_{i=1}^{n} i x_i^2'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = True

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
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        idx = np.arange(1, self.ndim + 1)
        return np.sum(idx * x ** 2)


class SumPower(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{SumPower}}(\mathbf{x}) = \sum_{i=1}^{n} \lvert x_i \rvert^{i+1}

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(\mathbf{x}) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """

    name = "SumPower Function"
    latex_formula = r'f_{\text{SumPower}}(\mathbf{x}) = \sum_{i=1}^{n} \lvert x_i \rvert^{i+1}'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = True

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
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        exponents = np.arange(2, self.ndim + 2)
        return np.sum(np.abs(x) ** exponents)


class StyblinskiTang(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{StyblinskiTang}}(\mathbf{x}) = \frac{1}{2}\sum_{i=1}^{n}\left(x_i^4 - 16x_i^2 + 5x_i\right)

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-5, 5]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`x_i^* \approx -2.903534` for :math:`i = 1, ..., n`
    """

    name = "Styblinski-Tang Function"
    latex_formula = r'f(\mathbf{x}) = \frac{1}{2}\sum_{i=1}^{n}\left(x_i^4 - 16x_i^2 + 5x_i\right)'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-5, 5], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'x_i^*\approx -2.903534, f(x^*)\approx -39.1661657d'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-5., 5.] for _ in range(self.dim_default)]))
        self.x_global = np.full(self.ndim, -2.903534018185960)
        self.f_global = -39.16616570377142 * self.ndim

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return 0.5 * np.sum(x ** 4 - 16.0 * x ** 2 + 5.0 * x)


class Schwefel221(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Schwefel2.21}}(\mathbf{x}) = \max_{i} \lvert x_i \rvert

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-100, 100]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(\mathbf{x}) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """

    name = "Schwefel 2.21 Function"
    latex_formula = r'f(\mathbf{x}) = \max_i \lvert x_i \rvert'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = False

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = False

    modality = False

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return np.max(np.abs(x))


class Schwefel222(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Schwefel2.22}}(\mathbf{x}) = \sum_{i=1}^{n}\lvert x_i \rvert + \prod_{i=1}^{n}\lvert x_i \rvert

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-10, 10]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(\mathbf{x}) = 0` for :math:`x_i = 0` for :math:`i = 1, ..., n`
    """

    name = "Schwefel 2.22 Function"
    latex_formula = r'f(\mathbf{x}) = \sum_{i=1}^{n}\lvert x_i \rvert + \prod_{i=1}^{n}\lvert x_i \rvert'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-10, 10], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
    continuous = True
    linear = False
    convex = True
    unimodal = True
    separable = True

    differentiable = False
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
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        ax = np.abs(x)
        return np.sum(ax) + np.prod(ax)


class Schwefel(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Schwefel2.26}}(\mathbf{x}) = 418.9829n - \sum_{i=1}^{n}x_i\sin\left(\sqrt{\lvert x_i \rvert}\right)

    Here, :math:`n` represents the number of dimensions and :math:`x_i \in [-500, 500]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(\mathbf{x}) = 0` for :math:`x_i = 420.968746` for :math:`i = 1, ..., n`
    """

    name = "Schwefel Function"
    latex_formula = r'f(\mathbf{x}) = 418.9829n - \sum_{i=1}^{n}x_i\sin\left(\sqrt{\lvert x_i \rvert}\right)'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-500, 500], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(420.968746, ...,420.968746) = 0'
    continuous = True
    linear = False
    convex = False
    unimodal = False
    separable = True

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = False

    modality = True

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-500., 500.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.full(self.ndim, 420.968746)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return 418.9829 * self.ndim - np.sum(x * np.sin(np.sqrt(np.abs(x))))


class Step(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Step}}(\mathbf{x}) = \sum_{i=1}^{n}\lfloor \lvert x_i \rvert \rfloor
    """

    name = "Step Function"
    latex_formula = r'f(\mathbf{x}) = \sum_{i=1}^{n}\lfloor \lvert x_i \rvert \rfloor'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
    continuous = False
    linear = False
    convex = True
    unimodal = True
    separable = True

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = False

    modality = False

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return np.sum(np.floor(np.abs(x)))


class Step2(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Step2}}(\mathbf{x}) = \sum_{i=1}^{n}\lfloor x_i + 0.5 \rfloor^2
    """

    name = "Step2 Function"
    latex_formula = r'f(\mathbf{x}) = \sum_{i=1}^{n}\lfloor x_i + 0.5 \rfloor^2'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(-0.5, -0.5, ...,-0.5) = 0'
    continuous = False
    linear = False
    convex = True
    unimodal = True
    separable = True

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = False

    modality = False

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.full(self.ndim, -0.5)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return np.sum(np.floor(x + 0.5) ** 2)


class Step3(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f_{\text{Step3}}(\mathbf{x}) = \sum_{i=1}^{n}\lfloor x_i^2 \rfloor
    """

    name = "Step3 Function"
    latex_formula = r'f(\mathbf{x}) = \sum_{i=1}^{n}\lfloor x_i^2 \rfloor'
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
    continuous = False
    linear = False
    convex = True
    unimodal = True
    separable = True

    differentiable = False
    scalable = True
    randomized_term = False
    parametric = False

    modality = False

    def __init__(self, ndim=None, bounds=None):
        super().__init__()
        self.dim_changeable = True
        self.dim_default = 2
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        return np.sum(np.floor(x ** 2))


class Schaffer(Benchmark):
    r"""
    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions For Global Optimization Problems
    Int. Journal of Mathematical Modelling and Numerical Optimisation, 2013, 4, 150-194.

    .. math::

        f(\mathbf{x}) = 0.5 + \frac{\sin^2\left(\sqrt{\sum_{i=1}^n x_i^2}\right) - 0.5}{\left(1 + 0.001\sum_{i=1}^n x_i^2\right)^2}
    """

    name = "Schaffer Function"
    latex_formula = (r'f(\mathbf{x}) = 0.5 + \frac{\sin^2\left(\sqrt{\sum_{i=1}^n x_i^2}\right) - 0.5}'
                     r'{\left(1 + 0.001\sum_{i=1}^n x_i^2\right)^2}')
    latex_formula_dimension = r'd \in \mathbb{N}_{+}^{*}'
    latex_formula_bounds = r'x_i \in [-100, 100], \forall i \in \llbracket 1, d\rrbracket'
    latex_formula_global_optimum = r'f(0, 0, ...,0) = 0'
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
        self.check_ndim_and_bounds(ndim, bounds, np.array([[-100., 100.] for _ in range(self.dim_default)]))
        self.f_global = 0.0
        self.x_global = np.zeros(self.ndim)

    def evaluate(self, x, *args):
        self.check_solution(x)
        self.n_fe += 1
        x = np.asarray(x)
        sum_sq = np.sum(x ** 2)
        return 0.5 + (np.sin(np.sqrt(sum_sq)) ** 2 - 0.5) / (1.0 + 0.001 * sum_sq) ** 2
