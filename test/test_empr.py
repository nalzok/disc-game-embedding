from main.embedding import DiscGameEmbed, EmpiricalInput, EmpiricalSupport
import numpy as np


def poly(order):
    def x_n(x):
        return x**order

    return x_n


basis = []
order = 5
for i in range(order):
    basis.append(poly(i))


xmin = np.array([0])
xmax = np.array([4])

sample = np.array([1, 2, 3])
f = np.array([[0, 1, -1], [-1, 0, 2], [1, -2, 0]])

support = EmpiricalSupport(sample)
payoff = EmpiricalInput(f, support)
game = DiscGameEmbed(payoff, basis)
game.SolveEmbedding()

for (x, y) in [(1, 2), (2, 3), (1, 3)]:
    print("f_hat(x, y) =", game.EvalSumDiscGame(2, x, y))
