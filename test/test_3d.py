import numpy as np
from main.embedding import DiscGameEmbed
from main.embedding import DiscGameEmbed, FunctionalInput, FunctionalSupport


def constant(x):
    return np.array(1)


def b1(x):
    return x[0]**2


def b2(x):
    return x[1]**2


def b3(x):
    return x[2]**2

basis = [constant, b1, b2]

def g(x, y):
    ## add x^T Z y for some skew symmetric matrix Z
    return np.sum(x**2) - np.sum(y**2)

xmin = np.array([0, 0, 0])
xmax = np.array([1, 1, 1])

def pi(x):
    return np.array(1)


support = FunctionalSupport(pi, xmin, xmax)
payoff = FunctionalInput(g, support)
game = DiscGameEmbed(payoff, basis)
game.SolveEmbedding()

for (x, y) in [(0.5, 0.3), (0.123, 0.456)]:
    print("g(x, y) - g_hat(x, y) =", g(x, y) - game.EvalSumDiscGame(2, x, y))
