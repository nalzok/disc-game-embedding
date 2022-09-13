import math
from main.embedding import DiscGameEmbed


def f(x, y):
    return x**2 * y - y**2 * x + x**2 - y**2 + x**3 * y - y**3 * x + x - y


xmin = 0
xmax = 1


def pi(x):
    return 1


# Trigo
def g(x, y):
    return math.sin(x) * math.cos(y) - math.sin(y) * math.cos(x)


basis = [math.sin, math.cos]

Game2 = DiscGameEmbed(basis, "quad", g, xmin, xmax, pi)
Game2.SolveEmbedding()

# 2-d vector works for polynomial
