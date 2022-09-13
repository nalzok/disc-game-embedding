from main.DiskGameEmbedding import DiscGameEmbed


def poly(order):
    def x_n(x):
        return x**order

    return x_n


def f(x, y):
    # f(x, y) = -f(y, x)
    return x**2 * y - y**2 * x + x**2 - y**2 + x**3 * y - y**3 * x + x - y


xmin = 0
xmax = 1


def pi(x):
    return 1


basis = []
order = 5
for i in range(order):
    basis.append(poly(i))

Game1 = DiscGameEmbed(basis, "quad", f, xmin, xmax, pi)

Game1.GramSchmidt()
Game1.UpdateProjection()
Game1.UpdateEmbedding()

X = Game1.EvaluateDiscGame(0, 0.5, 0.3)
print(Game1.embed_coef)

# polynomial okay
