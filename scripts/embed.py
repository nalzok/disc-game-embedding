from pathlib import Path

import numpy as np
import scipy.linalg as la
import click


def embed(F):
    n, n = F.shape
    T, Q = la.schur(F)
    diagT = np.diagonal(T, offset=-1)
    even = np.arange(0, n if n % 2 == 0 else n - 1, 2)
    odd = np.arange(1, n, 2)
    diagSqrtD = np.zeros(n)
    diagSqrtD[even] = diagSqrtD[odd] = np.sqrt(np.abs(diagT[even]))
    sqrtD = np.diag(diagSqrtD)

    # Sanity check
    R = np.zeros((n, n))
    sign = np.sign(diagT[even])
    R[even, odd] = -sign
    R[odd, even] = sign
    np.testing.assert_allclose(F, Q @ sqrtD @ R @ sqrtD @ Q.T, atol=0.001)

    embedding = sqrtD @ Q.T
    return embedding


@click.command()
@click.option("--payoff", type=click.Path(path_type=Path), required=True)
@click.option("--embedding", type=click.Path(path_type=Path), required=True)
def cli(payoff: Path, embedding: Path):
    F = np.load(payoff)
    ebd = embed(F)
    np.save(embedding, ebd)


if __name__ == "__main__":
    cli()
