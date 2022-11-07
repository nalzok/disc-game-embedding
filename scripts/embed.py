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
    reorder = np.arange(n)

    # Make sure each 2x2 block have sign [[0, +], [-, 0]]
    reorder[even] += diagT[even] > 0
    reorder[odd] -= diagT[even] > 0
    Q = Q[:, reorder]

    diagSqrtD = np.zeros(n)
    diagSqrtD[even] = diagSqrtD[odd] = np.sqrt(np.abs(diagT[even]))
    sqrtD = np.diag(diagSqrtD)

    # Sanity check
    R = np.zeros((n, n))
    R[even, odd] = 1
    R[odd, even] = -1
    np.testing.assert_allclose(F, Q @ sqrtD @ R @ sqrtD @ Q.T, atol=0.001)

    embedding = sqrtD @ Q.T
    eigen = np.diag(sqrtD)**2
    return embedding, eigen


@click.command()
@click.option("--payoff", type=click.Path(path_type=Path), required=True)
@click.option("--embedding", type=click.Path(path_type=Path), required=True)
@click.option("--eigen", type=click.Path(path_type=Path), required=True)
def cli(payoff: Path, embedding: Path, eigen: Path):
    F = np.load(payoff)
    embedding_, eig = embed(F)
    np.save(embedding, embedding_.T)
    np.save(eigen, eig)


if __name__ == "__main__":
    cli()
