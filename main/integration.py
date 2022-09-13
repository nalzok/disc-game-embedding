import numpy as np
import scipy.integrate as integrate


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
def inner_product_omega(f1, f2, pi_x, xmin, xmax, xsample, method):
    if method == "quad":
        return integrate_omega_quad(f1, f2, pi_x, xmin, xmax)
    # can define other integration method: e.g empirical measure/ normal, user defined measure etc
    elif method == "empirical":
        return integrate_omega_empirical(f1, f2, xsample)
    else:
        raise Exception("Invalid integrating method")


def inner_product_omega_omega(f, f1, f2, pi_x, xmin, xmax, method="quad"):
    if method == "quad":
        return integrate_omega_omega_quad(f, f1, f2, pi_x, xmin, xmax)
    else:
        raise Exception("Invalid integrating method")


# numerical integration
def integrate_omega_quad(f1, f2, pi_x, xmin, xmax):
    integrand = decorator_omega(f1, f2, pi_x)
    int_range = []
    if type(xmin) == int:
        int_range = [[xmin, xmax]]
    else:
        for i in range(len(xmin)):
            int_range.append([xmin[i], xmax[i]])
    return integrate.nquad(integrand, int_range)[0]


def integrate_omega_omega_quad(f, f1, f2, pi_x, xmin, xmax):
    integrand = decorator_omega_omega(f, f1, f2, pi_x)
    int_range = []
    if type(xmin) == int:
        int_range = [[xmin, xmax], [xmin, xmax]]
    else:
        ## work twice. 1st time for x, second time for y
        for i in range(len(xmin)):
            int_range.append([xmin[i], xmax[i]])
        for i in range(len(xmin)):
            int_range.append([xmin[i], xmax[i]])

    I = integrate.nquad(integrand, int_range)[0]
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
    def f_sum(x):
        temp = 0
        for i in range(len(f_list)):
            temp += coef_v[i] * f_list[i](x)
        return temp

    return f_sum


def function_scale(f, factor):
    def f_scale(x):
        return factor * f(x)

    return f_scale
