#!/usr/bin/env python
# Created by "Thieu" at 17:38, 30/07/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
import opfunu


def _assert_common(problem, ndim):
    x = np.ones(ndim)
    result = problem.evaluate(x)
    assert type(result) == np.float64
    assert isinstance(problem, opfunu.name_based.Benchmark)
    assert isinstance(problem.lb, np.ndarray)
    assert len(problem.lb) == ndim
    assert problem.bounds.shape[0] == ndim
    assert len(problem.x_global) == ndim


def test_Penalized1_results():
    ndim = 17
    _assert_common(opfunu.name_based.Penalized1(ndim=ndim), ndim)


def test_Penalized2_results():
    ndim = 17
    _assert_common(opfunu.name_based.Penalized2(ndim=ndim), ndim)
