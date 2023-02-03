from hashlib import sha256
import json

from flask import request, render_template, redirect, url_for
import numpy as np

from app import app
from app.embed import embed_algo

datastore = {}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@app.route('/')
def index():
    hexdigest = request.args.get("hexdigest")
    scaling = request.args.get("scaling")
    color_by = request.args.get("color_by")
    return render_template("index.html", hexdigest=hexdigest, scaling=scaling, color_by=color_by)


@app.route('/api/v1/embed', methods=['GET', 'POST'])
def embed():
    if request.method == "POST":
        if request.files["features"] and request.files["payoff"]:
            features = request.files["features"]
            payoff = request.files["payoff"]

            X = np.load(features)
            F = np.load(payoff)
            embedding, eigen = embed_algo(F)

            m = sha256()
            m.update(F.tobytes())
            m.update(X.tobytes())
            hexdigest = m.hexdigest()

            result = {
                "features": X,
                "payoff": F,
                "embedding": embedding,
                "eigen": eigen,
            }

            datastore[hexdigest] = json.dumps(result, cls=NumpyEncoder)

        elif request.form["hexdigest"]:
            hexdigest = request.form["hexdigest"]

        else:
            raise ValueError("Neither features/payoff nor hexdigest is supplied")

        scaling = request.form["scaling"]
        color_by = request.form["color_by"]
        return redirect(url_for("index", hexdigest=hexdigest, scaling=scaling, color_by=color_by))

    hexdigest = request.args.get("hexdigest")
    return datastore[hexdigest]
