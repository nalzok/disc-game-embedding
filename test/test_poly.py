from main.embedding import DiscGameEmbed, FunctionalInput, FunctionalSupport
import numpy as np


def poly(order):
    def x_n(x):
        return x**order

    return x_n


basis = []
order = 5
for i in range(order):
    basis.append(poly(i))


def f(x, y):
    # f(x, y) = -f(y, x)
    return x**2 * y - y**2 * x + x**2 - y**2 + x**3 * y - y**3 * x + x - y


xmin = np.array([0])
xmax = np.array([1])


def pi(x):
    return np.array(1)


support = FunctionalSupport(pi, xmin, xmax)
payoff = FunctionalInput(f, support)
game = DiscGameEmbed(payoff, basis)
game.SolveEmbedding()

for (x, y) in [(0.5, 0.3), (0.123, 0.456)]:
    print("f(x, y) - f_hat(x, y) =", f(x, y) - game.EvalSumDiscGame(2, x, y))
