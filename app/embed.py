import numpy as np
import scipy.linalg as la


def embed_algo(F):
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
    embedding = embedding.T
    eigen = np.diag(sqrtD)**2
    return embedding, eigen
