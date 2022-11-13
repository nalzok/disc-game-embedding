import numpy as np
import jax
import optax
from scipy import linalg


def MC_GD(M, Omega, r, maxIter=1000, tol=1e-3):
    """
    References
    ----------
    .. [1] Chen, Ji, Xiaodong Li, and Zongming Ma. "Nonconvex matrix completion with linearly parameterized factors." Journal of Machine Learning Research 23.207 (2022): 1-35.
    .. [2] Sun, Ruoyu, and Zhi-Quan Luo. "Guaranteed matrix completion via non-convex factorization." IEEE Transactions on Information Theory 62.11 (2016): 6535-6579.
    .. [3] Chi, Yuejie. "Low-rank matrix completion [lecture notes]." IEEE Signal Processing Magazine 35.5 (2018): 178-181.

    Parameters:
    ----------
    M: array
        Observed matrix.
    Omega: array
        Indicator matrix (0 = unobserved; 1 = observed).
    r: even integer
        Desired rank.
    maxIter: integer, optional
        Maximum allowed iteration. Default is 1000.
    tol: float, optional
        Absolute tolerances. Default is 1e-3.

    Returns:
    ----------
    M_rec: array
        Recovered matrix.
    """
    if r % 2 != 0:
        raise ValueError("r must be an even number")

    N = len(M)

    # empirical sample rate
    p = np.sum(np.sum(Omega)) / N**2

    @jax.jit
    def step(params, opt_state):
        @jax.value_and_grad
        def objective(params):
            A, B = params
            recon = np.sum((Omega * (A @ B.T - B @ A.T - M)) ** 2)
            regularizer1 = np.sum((A.T @ A - B.T @ B) ** 2)
            regularizer2 = np.sum((A.T @ B + B.T @ A) ** 2)
            return recon / (2 * p) + regularizer1 / 4 + regularizer2 / 4

        loss, grads = objective(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state, loss

    # Initialization
    U, Q = linalg.schur(M / p)
    A = np.zeros((N, r // 2))
    B = np.zeros((N, r // 2))
    for i in range(r // 2):
        A[:, i] = np.sqrt(np.abs(U[2 * i + 1, 2 * i])) * Q[:, 2 * i + 1]
        B[:, i] = np.sqrt(np.abs(U[2 * i + 1, 2 * i])) * Q[:, 2 * i]

    # key = jax.random.PRNGKey(42)
    # key_A, key_B = jax.random.split(key)
    # A = jax.random.normal(key_A, (N, r // 2))
    # B = jax.random.normal(key_B, (N, r // 2))

    # Optimization
    schedule = optax.warmup_cosine_decay_schedule(
      init_value=0.0,
      peak_value=0.1,
      warmup_steps=maxIter // 20,
      decay_steps=maxIter,
      end_value=0.0,
    )
    optimizer = optax.chain(
      optax.clip(1.0),
      optax.adamw(learning_rate=schedule),
    )
    params = A, B
    opt_state = optimizer.init(params)

    # gradient
    for i in range(maxIter):
        params, opt_state, obj = step(params, opt_state)
        print("iter: %d, obj: %f" % (i, obj))

    return A @ B.T - B @ A.T


def MC_SVT(M, Omega, t=None, delta_t=2, maxIter=1000, tol=1e-2):
    """
    References
    ----------
    .. [1] Cai, Jian-Feng, Emmanuel J. Candès, and Zuowei Shen. "A singular value thresholding algorithm for matrix completion." SIAM Journal on optimization 20.4 (2010): 1956-1982.

    Parameters:
    ----------
    M: array
        Observed matrix.
    Omega: array
        Indicator matrix (0:observed;1:observed).
    t: float,optimal
        Singular value threshood. Default is 2*N.

    delta_t: float, optional
        Step size. Default is 2. Try larger values if it's too slow. Try smaller values if it doesn't converge.
    maxIter: integer, optional
        Maximum allowed iteration. Default is 1000.
    tol: float, optional
        Absolute tolerances. Default is 1e-2.

    Returns:
    ----------
    M_rec: array
        Recovered matrix.

    """

    N = len(M)
    if t == None:
        t = 2 * N

    X = Omega * M
    Y = np.zeros((N, N))
    iters = 0
    epsilon = 1e5

    while epsilon > tol and iters < maxIter:
        U, S, VT = np.linalg.svd(Y)
        S = S - t
        ind = S > 0
        S = S * ind
        X = U @ np.diag(S) @ VT
        Y = Y + delta_t * Omega * (M - X)
        epsilon = np.linalg.norm(Omega * (M - X), "fro") / np.linalg.norm(
            Omega * M, "fro"
        )
        iters = iters + 1
        if iters == 1 or iters % 50 == 0 or epsilon < tol:
            print("iter: %d, err: %f" % (iters, epsilon))

    return X
