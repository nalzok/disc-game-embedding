from ast import literal_eval
from flask import request, jsonify
import numpy as np

from app import app
from main.embedding import DiscGameEmbed, EmpiricalInput, EmpiricalSupport


def poly(order):
    def x_n(x):
        return x**order

    return x_n


basis = []
order = 5
for i in range(order):
    basis.append(poly(i))


@app.route('/api/v1/embed', methods=['GET', 'POST'])
def embed():
    # curl -X POST http://127.0.0.1:5000/api/v1/embed -H 'Content-Type: application/json' -d '{"sample": [1, 2, 3], "f": [[0, 1, -1], [-1, 0, 2], [1, -2, 0]], "test": [[1, 2], [2, 3]]}'
    content = request.json

    sample = np.array(literal_eval(repr(content["sample"])))
    f = np.array(literal_eval(repr(content["f"])))

    support = EmpiricalSupport(sample)
    payoff = EmpiricalInput(f, support)
    game = DiscGameEmbed(payoff, basis)
    game.SolveEmbedding()

    result = []
    for (x, y) in literal_eval(repr(content["test"])):
        result.append(game.EvalSumDiscGame(2, x, y))

    return jsonify({"result": result})
