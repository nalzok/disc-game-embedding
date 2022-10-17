from typing import Sequence, Union, cast
import math
import numpy as np
import scipy.linalg as la

from .integration import (
    UnaryCallable,
    BinaryCallable,
    FunctionalSupport,
    FunctionalInput,
    EmpiricalSupport,
    EmpiricalInput,
    inner_product_omega,
    inner_product_omega_omega,
    function_sum,
    function_scale,
)


class DiscGameEmbed:
    # constructor
    # sample should be of sorted of increasing value
    # f_sample is the matrix of samples f(xi, xj). It should be arranged by sorting xi, xj respectively.
    def __init__(
        self,
        payout: Union[FunctionalInput, EmpiricalInput],
        basis: Sequence[UnaryCallable],
    ):
        self.basis = basis

        # if abs(x) < machine_epsilon, x = 0
        self.epsilon = np.finfo(float).eps

        if isinstance(payout, FunctionalInput):
            self.method = "quad"
            self.f = payout.f
            self.support = payout.support
        elif isinstance(payout, EmpiricalInput):
            self.method = "empirical"
            self.f = payout.f
            self.support = payout.support
            assert (
                self.f.shape == (self.support.sample.size, self.support.sample.size)
            )
        else:
            raise ValueError(f"Unknown payout specification {type(payout)}")

    # Gram Schmidt
    # update basis_orthorgonal and gram_coef
    # gram_coef: row i = basis_orthogonal[i]'s coef
    def GramSchmidt(self):
        n = len(self.basis)
        if n == 0:
            raise Exception("There should be at least 1 basis function")
        self.basis_orthogonal = []
        self.gram_coef = np.zeros((n, n))
        row_idx_v = []
        for i in range(n):
            coef_v = np.zeros(n)
            coef_v[i] = 1
            for j in range(len(self.basis_orthogonal)):
                row_idx = row_idx_v[j]
                coef = inner_product_omega(
                    self.basis[i],
                    self.basis_orthogonal[j],
                    self.support,
                )
                coef_v -= coef * self.gram_coef[row_idx]
            # create the orthogonal basis
            ortho_basis = function_sum(self.basis, coef_v)
            # check linear independence
            norm = inner_product_omega(
                ortho_basis,
                ortho_basis,
                self.support,
            )
            norm = math.sqrt(norm)

            if norm > self.epsilon:
                self.gram_coef[i] = coef_v / norm
                self.basis_orthogonal.append(function_scale(ortho_basis, 1 / norm))
                row_idx_v.append(i)

    def UpdateProjection(self):
        n = len(self.basis)
        B = np.zeros((n, n))
        if isinstance(self.support, FunctionalSupport):
            self.f = cast(BinaryCallable, self.f)
            for i in range(n):
                for j in range(n):
                    B[i][j] = inner_product_omega_omega(
                        self.f,
                        self.basis[i],
                        self.basis[j],
                        self.support,
                    )
        elif isinstance(self.support, EmpiricalSupport):
            assert isinstance(self.f, np.ndarray) and len(self.f.shape) == 2
            m = len(self.support.sample)
            C = np.empty((m, n))
            for i in range(m):
                for j in range(n):
                    C[i][j] = self.basis[j](self.support.sample[i])
            B = C.T @ self.f @ C / (m**2)
        else:
            raise ValueError(f"Unknown support specification {type(self.support)}")

        self.projection = self.gram_coef @ B @ self.gram_coef.T

        if self.projection.shape[0] % 2 == 1:
            np.pad(
                self.projection, ((0, 1), (0, 1)), mode="constant", constant_values=0
            )

    def UpdateEmbedding(self):
        # Schur decomp
        T, Q = la.schur(self.projection)
        eigen = np.zeros(T.shape[0] // 2)
        # get lambda and drop 0 eigenvalues
        for i in range(T.shape[0] // 2):
            eigen[i] = T[2 * i, 2 * i + 1]
        eigen = eigen[abs(eigen) > self.epsilon * T.shape[0]]

        # update rank
        self.rank = len(eigen)

        # update lambda
        self.v_lambda = np.sqrt(abs(eigen))

        # update embedding
        sort_idx = np.argsort(np.abs(eigen))
        m_coef = np.zeros(Q.shape)
        for i, idx in enumerate(sort_idx):
            lambda_ = eigen[idx]
            # switch the rows if the top right corner of the block diagonal matrix is negative
            if lambda_ < 0:
                temp = math.sqrt(-1 * lambda_)
                m_coef[2 * i] = temp * Q.T[2 * idx + 1]
                m_coef[2 * i + 1] = temp * Q.T[2 * idx]
            else:
                temp = math.sqrt(lambda_)
                m_coef[2 * i] = temp * Q.T[2 * idx]
                m_coef[2 * i + 1] = temp * Q.T[2 * idx + 1]

        n = len(self.basis_orthogonal)
        self.embed_coef_ortho = m_coef[:, :n]
        self.embed_coef = self.embed_coef_ortho @ self.gram_coef

        # create embedding functions
        self.discgame_embedding = []
        for i in range(self.rank):
            x = function_sum(self.basis, self.embed_coef[2 * i])
            y = function_sum(self.basis, self.embed_coef[2 * i + 1])
            self.discgame_embedding.append((x, y))

    def SolveEmbedding(self):
        self.GramSchmidt()
        self.UpdateProjection()
        self.UpdateEmbedding()

    def EvaluateDiscGame(self, i, x, y):
        if i > self.rank:
            raise Exception(
                f"please enter a proper index of disc games. The index should be less than the rank = {self.rank}"
            )

        if i < 0:
            raise Exception("The index of disc game embedding should be non-negative.")

        f1 = self.discgame_embedding[i - 1][0]
        f2 = self.discgame_embedding[i - 1][1]

        return f1(x) * f2(y) - f1(y) * f2(x)

    def EvalSumDiscGame(self, i, x, y):
        value = 0
        for i in range(i):
            value += self.EvaluateDiscGame(i + 1, x, y)
        return value
