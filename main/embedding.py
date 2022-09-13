import math
import numpy as np
import scipy.linalg as la

from .integration import (
    inner_product_omega,
    inner_product_omega_omega,
    function_sum,
    function_scale,
)


# if abs(x) < machine_epsilon, x = 0
machine_epsilon = np.finfo(float).eps


class DiscGameEmbed:
    # constructor
    # sample should be of sorted of increasing value
    # f_sample is the matrix of samples f(xi, xj). It should be arranged by sorting xi, xj respectively.
    def __init__(
        self, basis_list, measure, f=0, xmin=0, xmax=0, pi_x=0, sample=[0], f_sample=0
    ):
        #'set of orginal basis
        #'@ param basis set of orginal basis
        self.basis = basis_list
        self.f = f
        self.xmin = xmin
        self.xmax = xmax
        self.pi_x = pi_x
        self.basis_orthogonal = []
        self.gram_coef = np.empty(shape=(0, len(basis_list)))
        self.projection = np.empty(shape=(0, 0))
        self.discgame_embedding = []
        self.rank = 0
        self.embed_coef = np.empty(shape=(0, 0))
        self.embed_coef_ortho = np.empty(shape=(0, 0))
        self.v_lambda = np.empty(shape=(0, 0))
        self.method = measure
        self.sample = sample
        self.f_sample = f_sample

    # GramSchmidth
    # update basis_orthorgonal and gram_coef
    # gram_coef: row i = basis_orthogonal[i]'s coef
    def GramSchmidt(self):
        if len(self.basis) == 0:
            raise Exception("There should be at least 1 function basis ")
        self.basis_orthogonal = []
        self.gram_coef = np.empty(shape=(0, len(self.basis)))
        n = len(self.basis)
        row_idx_v = []
        for i in range(n):
            coef_v = np.zeros(n)
            coef_v[i] = 1
            for j in range(len(self.basis_orthogonal)):
                row_idx = row_idx_v[j]
                coef = inner_product_omega(
                    self.basis[i],
                    self.basis_orthogonal[j],
                    self.pi_x,
                    self.xmin,
                    self.xmax,
                    self.sample,
                    self.method,
                )
                coef_v -= coef * self.gram_coef[row_idx]
            # create the orthogonal basis
            ortho_basis = function_sum(self.basis, coef_v)
            # check linear independce
            norm = inner_product_omega(
                ortho_basis,
                ortho_basis,
                self.pi_x,
                self.xmin,
                self.xmax,
                self.sample,
                self.method,
            )
            norm = math.sqrt(norm)

            if norm > machine_epsilon:
                self.gram_coef = np.vstack((self.gram_coef, 1 / norm * coef_v))
                self.basis_orthogonal.append(function_scale(ortho_basis, 1 / norm))
                row_idx_v.append(i)

    def UpdateProjection(self):
        n = len(self.basis)
        B = np.zeros((n, n))
        if self.method == "quad":
            for i in range(n):
                for j in range(n):
                    B[i][j] = inner_product_omega_omega(
                        self.f,
                        self.basis[i],
                        self.basis[j],
                        self.pi_x,
                        self.xmin,
                        self.xmax,
                    )
        if self.method == "empirical":
            C = np.zeros((len(self.sample), len(self.basis)))
            for i in range(len(self.sample)):
                for j in range(len(self.basis)):
                    C[i][j] = self.basis[j](self.sample[i])
            B = 1 / ((len(self.sample)) ** 2) * C.T @ self.f_sample @ C
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
        eigen = eigen[abs(eigen) > machine_epsilon * T.shape[0]]

        # update rank
        self.rank = len(eigen)

        # update lambda
        self.v_lambda = np.sqrt(abs(eigen))

        # update embedding
        sort_idx = np.argsort(np.abs(eigen))
        m_coef = np.zeros(Q.shape)
        for i in range(len(sort_idx)):
            idx = sort_idx[i]
            _lambda = eigen[idx]
            temp = 0
            # switch the rows if the top right corner of the block diagonal matrix is negative
            if _lambda < 0:
                temp = math.sqrt(-1 * _lambda)
                m_coef[2 * i] = temp * Q.T[2 * idx + 1]
                m_coef[2 * i + 1] = temp * Q.T[2 * idx]
            else:
                temp = math.sqrt(_lambda)
                m_coef[2 * i] = temp * Q.T[2 * idx]
                m_coef[2 * i + 1] = temp * Q.T[2 * idx + 1]

        n = len(self.basis_orthogonal)
        self.embed_coef_ortho = m_coef[:, 0:n]
        self.embed_coef = self.embed_coef_ortho @ self.gram_coef

        # create embedding functions
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
                "please enter a proper index of disc games. The index should be less than the rank"
            )

        if i < 0:
            raise Exception("The index of disc game embedding should be non-negative.")

        f1 = self.discgame_embedding[i - 1][0]
        f2 = self.discgame_embedding[i - 1][1]

        return f1(x) * f2(y) - f1(y) * f2(x)

    def EvalSumDiscGame(self, i, x, y):
        temp = 0
        for i in range(i):
            temp += self.EvaluateDiscGame(i + 1, x, y)
        return temp
