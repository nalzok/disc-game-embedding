from itertools import product
import numpy as np
from main.embedding import DiscGameEmbed, EmpiricalInput, EmpiricalSupport


def poly(order):
    def x_n(x):
        return x**order

    return x_n


basis = []
order = 5
for i in range(order):
    basis.append(poly(i))


@np.vectorize
def f(x, y):
    return x**2*y -y**2*x +x**2 -y**2 +x**3*y -y**3*x +x -y


n = 10
rng = np.random.default_rng(42)
X = rng.uniform(size=n)
F = f(X[:, np.newaxis], X)

support = EmpiricalSupport(X)
payoff = EmpiricalInput(F, support)
game = DiscGameEmbed(payoff, basis)
game.SolveEmbedding()

max_error = 0
for x, y in product(X, repeat=2):
    max_error = max(max_error, abs(f(x, y) - game.EvalSumDiscGame(order // 2, x, y)))

print(f"{max_error = }")
