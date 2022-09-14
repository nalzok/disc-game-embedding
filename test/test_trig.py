import numpy as np
from main.embedding import DiscGameEmbed, FunctionalInput, FunctionalSupport


basis = [np.sin, np.cos]

# Trigonometric
def g(x, y):
    return np.sin(x) * np.cos(y) - np.sin(y) * np.cos(x)

xmin = np.array([0])
xmax = np.array([1])

def pi(x):
    return np.array(1)

support = FunctionalSupport(pi, xmin, xmax)
payoff = FunctionalInput(g, support)
game = DiscGameEmbed(payoff, basis)
game.SolveEmbedding()

for (x, y) in [(0.5, 0.5), (0.123, 0.456)]:
    print("g(x, y) - g_hat(x, y) =", g(x, y) - game.EvalSumDiscGame(1, x, y))
