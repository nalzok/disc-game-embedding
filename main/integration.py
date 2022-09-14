from typing import Callable, Union
from dataclasses import dataclass
import numpy as np
import scipy.integrate as integrate


UnaryCallable = Callable[[np.ndarray], np.ndarray]
BinaryCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class FunctionalSupport:
    pi_x: UnaryCallable
    xmin: np.ndarray
    xmax: np.ndarray


@dataclass
class FunctionalInput:
    f: BinaryCallable
    support: FunctionalSupport


@dataclass
class EmpiricalSupport:
    sample: np.ndarray


@dataclass
class EmpiricalInput:
    f: np.ndarray
    support: EmpiricalSupport


Support = Union[FunctionalSupport, EmpiricalSupport]


# decorator for integration
def decorator_omega(f1, f2, pi_x):
    def wrapper_omega(*args):
        x = np.array(args)
        return f1(x) * f2(x) * pi_x(x)

    return wrapper_omega


def decorator_omega_omega(f, f1, f2, pi_x):
    def wrapper_omega_omega(*args):
        n = len(args) // 2
        x = np.array(args[0:n])
        y = np.array(args[n : 2 * n])
        return f(x, y) * f1(x) * f2(y) * pi_x(x) * pi_x(y)

    return wrapper_omega_omega


# inner product
def inner_product_omega(f1: UnaryCallable, f2: UnaryCallable, support: Support):
    if isinstance(support, FunctionalSupport):
        return integrate_omega_quad(f1, f2, support.pi_x, support.xmin, support.xmax)
    # can define other integration method: e.g empirical measure/ normal, user defined measure etc
    elif isinstance(support, EmpiricalSupport):
        return integrate_omega_empirical(f1, f2, support.sample)
    else:
        raise ValueError("Invalid integrating method")


def inner_product_omega_omega(
    f: BinaryCallable, f1: UnaryCallable, f2: UnaryCallable, support: Support
):
    if isinstance(support, FunctionalSupport):
        return integrate_omega_omega_quad(
            f, f1, f2, support.pi_x, support.xmin, support.xmax
        )
    else:
        raise ValueError("Invalid integrating method")


# numerical integration
def integrate_omega_quad(f1, f2, pi_x, xmin, xmax):
    integrand = decorator_omega(f1, f2, pi_x)
    int_range = []
    for min_, max_ in zip(xmin, xmax):
        int_range.append((min_, max_))

    I, _ = integrate.nquad(integrand, int_range)
    return I


def integrate_omega_omega_quad(f, f1, f2, pi_x, xmin, xmax):
    integrand = decorator_omega_omega(f, f1, f2, pi_x)
    int_range = []
    ## work twice. 1st time for x, second time for y
    for min_, max_ in zip(xmin, xmax):
        int_range.append((min_, max_))
    for min_, max_ in zip(xmin, xmax):
        int_range.append((min_, max_))

    I, _ = integrate.nquad(integrand, int_range)
    return I


# Empirical measure
def integrate_omega_empirical(f1, f2, xsample):
    n = len(xsample)
    inner_prod = 0
    for i in range(n):
        inner_prod += f1(xsample[i]) * f2(xsample[i])
    return inner_prod / n


# function creation
def function_sum(f_list, coef_v):
    def f_sum(x: np.ndarray) -> np.ndarray:
        total = coef_v[0] * f_list[0](x)
        for f, v in zip(f_list[1:], coef_v[1:]):
            total += v * f(x)
        return total

    return f_sum


def function_scale(f, factor):
    def f_scale(x):
        return factor * f(x)

    return f_scale
