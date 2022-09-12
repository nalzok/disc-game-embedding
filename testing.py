# -*- coding: utf-8 -*-
"""

@author: patrick
"""

import math
import numpy as np
from DiskGameEmbedding import DiscGameEmbed

def poly(order):
    def x_n(x):
        return x**order
    return x_n

def f(x,y):
    return x**2*y-y**2*x+x**2-y**2 + x**3*y-y**3*x + x-y

xmin = 0
xmax = 1

def pi(x):
    return 1

basis = []
order = 5
for i in range(order):
    basis.append(poly(i))
    
Game1 = DiscGameEmbed(basis,"quad", f, xmin, xmax, pi)

Game1.GramSchmidt()
Game1.UpdateProjection()
Game1.UpdateEmbedding()

X = Game1.EvaluateDiscGame(0, 0.5, 0.3)

#polynomial okay

#Trigo
def g(x,y):
    return math.sin(x)*math.cos(y) - math.sin(y)*math.cos(x)


basis = [math.sin, math.cos]

Game2 = DiscGameEmbed(basis,"quad", g, xmin, xmax, pi)
Game2.SolveEmbedding()

#2-d vector works for polynomial
def g(x,y):
    return np.sum(x**2)-np.sum(y**2) ## add x^T Z y for some skew symmetric matrix Z

def b1(x):
    return x[0]**2

def b2(x):
    return x[1]**2


def constant(x):
    return 1

basis = [constant, b1, b2]
Game3 = DiscGameEmbed(basis,"quad", g, [0,0,0], [1,1,1], pi)
Game3.SolveEmbedding()
Game3.embed_coef

