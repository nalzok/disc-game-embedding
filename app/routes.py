from ast import literal_eval
from flask import request, jsonify
import numpy as np

from app import app
from main.embedding import DiscGameEmbed, EmpiricalInput, EmpiricalSupport


def poly_basis(order: int):
    def poly(i):
        return lambda x: x**i

    basis = [poly(i) for i in range(order)]
    return basis


@app.route('/api/v1/embed', methods=['GET', 'POST'])
def embed():
    content = request.json
    sample = np.array(literal_eval(repr(content["sample"])))
    M = np.array(literal_eval(repr(content["M"])))
    order = content["order"]
    test = literal_eval(repr(content["test"]))

    support = EmpiricalSupport(sample)
    payoff = EmpiricalInput(M, support)
    basis = poly_basis(order)

    game = DiscGameEmbed(payoff, basis)
    game.SolveEmbedding()

    result = []
    for x, y in test:
        result.append(game.EvalSumDiscGame(order // 2, x, y))

    return jsonify({"result": result})
