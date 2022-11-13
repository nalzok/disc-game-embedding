import numpy as np
import scipy
from main.completion import MC_GD, MC_SVT


pokem = np.load("embeddings/Pokemon/F.npy")
FM = pokem
N = len(FM)

# Random observation matrix
samp_rate = 0.5
np.random.seed(42)
Omega = scipy.stats.bernoulli.rvs(size=N * N, p=samp_rate).reshape(N, N)

for i in range(1, N):
    for j in range(N - 1):
        Omega[j, i] = Omega[i, j]

for i in range(N):
    Omega[i, i] = 1

assert np.testing.assert_allclose(Omega, Omega.T) is None

# observed matrix
OM = FM * Omega

M_rec = MC_GD(OM, Omega, 16, maxIter=1000, tol=1e-3)
# M_rec = MC_SVT(OM, Omega, delta_t=2, maxIter=1000, tol=1e-2)
