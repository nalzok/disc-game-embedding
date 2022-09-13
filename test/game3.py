import numpy as np
from main.embedding import DiscGameEmbed


def f(x, y):
    return x**2 * y - y**2 * x + x**2 - y**2 + x**3 * y - y**3 * x + x - y


xmin = 0
xmax = 1


def pi(x):
    return 1


def g(x, y):
    return np.sum(x**2) - np.sum(
        y**2
    )  ## add x^T Z y for some skew symmetric matrix Z


def b1(x):
    return x[0] ** 2


def b2(x):
    return x[1] ** 2


def constant(x):
    return 1


basis = [constant, b1, b2]
Game3 = DiscGameEmbed(basis, "quad", g, [0, 0, 0], [1, 1, 1], pi)
Game3.SolveEmbedding()
Game3.embed_coef
