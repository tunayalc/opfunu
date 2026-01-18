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


def test_Salomon_results():
    ndim = 17
    problem = opfunu.name_based.Salomon(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_Sphere_results():
    ndim = 17
    problem = opfunu.name_based.Sphere(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_SumSquares_results():
    ndim = 17
    problem = opfunu.name_based.SumSquares(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_SumPower_results():
    ndim = 17
    problem = opfunu.name_based.SumPower(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_StyblinskiTang_results():
    ndim = 17
    problem = opfunu.name_based.StyblinskiTang(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_Schwefel221_results():
    ndim = 17
    problem = opfunu.name_based.Schwefel221(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_Schwefel222_results():
    ndim = 17
    problem = opfunu.name_based.Schwefel222(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_Schwefel_results():
    ndim = 17
    problem = opfunu.name_based.Schwefel(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_Step_results():
    ndim = 17
    problem = opfunu.name_based.Step(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_Step2_results():
    ndim = 17
    problem = opfunu.name_based.Step2(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_Step3_results():
    ndim = 17
    problem = opfunu.name_based.Step3(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim


def test_Schaffer_results():
    ndim = 17
    problem = opfunu.name_based.Schaffer(ndim=ndim)
    _assert_common(problem, ndim)
    assert len(problem.x_global) == ndim
