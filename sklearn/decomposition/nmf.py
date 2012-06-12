# -*- coding: utf-8 -*-

""" Non-negative matrix factorization
"""
# Author: Olivier Mangin <olivier.mangin@inria.fr> (Beta-NMF implementation)
# Author: Vlad Niculae
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# Author: Chih-Jen Lin, National Taiwan University (original projected gradient
#     NMF implementation)
# Author: Anthony Di Franco (original Python and NumPy port)
# License: BSD


from __future__ import division

from ..base import BaseEstimator, TransformerMixin
from ..utils import atleast2d_or_csr, check_random_state
from ..utils.extmath import randomized_svd, safe_sparse_dot

import numpy as np
from scipy.optimize import nnls
import scipy.sparse as sp
import warnings


def _pos(x):
    """Positive part of a vector / matrix"""
    return (x >= 0) * x


def _neg(x):
    """Negative part of a vector / matrix"""
    neg_x = -x
    neg_x *= x < 0
    return neg_x


def norm(x):
    """Dot product-based Euclidean norm implementation

    See: http://fseoane.net/blog/2011/computing-the-vector-norm/
    """
    x = x.ravel()
    return np.sqrt(np.dot(x.T, x))


def _sparseness(x):
    """Hoyer's measure of sparsity for a vector"""
    sqrt_n = np.sqrt(len(x))
    return (sqrt_n - np.linalg.norm(x, 1) / norm(x)) / (sqrt_n - 1)


def check_non_negative(X, whom):
    X = X.data if sp.issparse(X) else X
    if (X < 0).any():
        raise ValueError("Negative values in data passed to %s" % whom)


def _initialize_nmf(X, n_components, variant=None, eps=1e-6,
                    random_state=None):
    """NNDSVD algorithm for NMF initialization.

    Computes a good initial guess for the non-negative
    rank k matrix approximation for X: X = WH

    Parameters
    ----------

    X: array, [n_samples, n_features]
        The data matrix to be decomposed.

    n_components:
        The number of components desired in the
        approximation.

    variant: None | 'a' | 'ar'
        The variant of the NNDSVD algorithm.
        Accepts None, 'a', 'ar'
        None: leaves the zero entries as zero
        'a': Fills the zero entries with the average of X
        'ar': Fills the zero entries with standard normal random variates.
        Default: None

    eps:
        Truncate all values less then this in output to zero.

    random_state: numpy.RandomState | int, optional
        The generator used to fill in the zeros, when using variant='ar'
        Default: numpy.random

    Returns
    -------

    (W, H):
        Initial guesses for solving X ~= WH such that
        the number of columns in W is n_components.

    Remarks
    -------

    This implements the algorithm described in
    C. Boutsidis, E. Gallopoulos: SVD based
    initialization: A head start for nonnegative
    matrix factorization - Pattern Recognition, 2008

    http://www.cs.rpi.edu/~boutsc/files/nndsvd.pdf
    """
    check_non_negative(X, "NMF initialization")
    if variant not in (None, 'a', 'ar'):
        raise ValueError("Invalid variant name")

    U, S, V = randomized_svd(X, n_components)
    W, H = np.zeros(U.shape), np.zeros(V.shape)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in xrange(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = _pos(x), _pos(y)
        x_n, y_n = _neg(x), _neg(y)

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if variant == "a":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif variant == "ar":
        random_state = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * random_state.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * random_state.randn(len(H[H == 0])) / 100)

    return W, H


def _nls_subproblem(V, W, H_init, tol, max_iter):
    """Non-negative least square solver

    Solves a non-negative least squares subproblem using the
    projected gradient descent algorithm.
    min || WH - V ||_2

    Parameters
    ----------
    V, W:
        Constant matrices

    H_init:
        Initial guess for the solution

    tol:
        Tolerance of the stopping condition.

    max_iter:
        Maximum number of iterations before
        timing out.

    Returns
    -------
    H:
        Solution to the non-negative least squares problem

    grad:
        The gradient.

    n_iter:
        The number of iterations done by the algorithm.

    """
    if (H_init < 0).any():
        raise ValueError("Negative values in H_init passed to NLS solver.")

    H = H_init
    WtV = safe_sparse_dot(W.T, V, dense_output=True)
    WtW = safe_sparse_dot(W.T, W, dense_output=True)

    # values justified in the paper
    alpha = 1
    beta = 0.1
    for n_iter in xrange(1, max_iter + 1):
        grad = np.dot(WtW, H) - WtV
        proj_gradient = norm(grad[np.logical_or(grad < 0, H > 0)])
        if proj_gradient < tol:
            break

        for inner_iter in xrange(1, 20):
            Hn = H - alpha * grad
            # Hn = np.where(Hn > 0, Hn, 0)
            Hn = _pos(Hn)
            d = Hn - H
            gradd = np.sum(grad * d)
            dQd = np.sum(np.dot(WtW, d) * d)
            # magic numbers whoa
            suff_decr = 0.99 * gradd + 0.5 * dQd < 0
            if inner_iter == 1:
                decr_alpha = not suff_decr
                Hp = H

            if decr_alpha:
                if suff_decr:
                    H = Hn
                    break
                else:
                    alpha *= beta
            elif not suff_decr or (Hp == Hn).all():
                H = Hp
                break
            else:
                alpha /= beta
                Hp = Hn

    if n_iter == max_iter:
        warnings.warn("Iteration limit reached in nls subproblem.")

    return H, grad, n_iter


class BaseNMF(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=None, init=None):
        self.n_components = n_components
        self.init = init

    def _init(self, X):
        n_samples, n_features = X.shape

        if self.init == 'nndsvd':
            W, H = _initialize_nmf(X, self.n_components)
        elif self.init == 'nndsvda':
            W, H = _initialize_nmf(X, self.n_components, variant='a')
        elif self.init == 'nndsvdar':
            W, H = _initialize_nmf(X, self.n_components, variant='ar')
        else:
            try:
                rng = check_random_state(self.init)
                W = rng.randn(n_samples, self.n_components)
                # we do not write np.abs(W, out=W) to stay compatible with
                # numpy 1.5 and earlier where the 'out' keyword is not
                # supported as a kwarg on ufuncs
                np.abs(W, W)
                H = rng.randn(self.n_components, n_features)
                np.abs(H, H)
            except ValueError:
                raise ValueError(
                    'Invalid init parameter: got %r instead of one of %r' %
                    (self.init, (None, 'nndsvd', 'nndsvda', 'nndsvdar',
                                 int, np.random.RandomState)))

        return W, H


class ProjectedGradientNMF(BaseNMF):
    """Non-Negative matrix factorization by Projected Gradient (NMF)

    Parameters
    ----------
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data the model will be fit to.

    n_components: int or None
        Number of components, if n_components is not set all components
        are kept

    init:  'nndsvd' |  'nndsvda' | 'nndsvdar' | int | RandomState
        Method used to initialize the procedure.
        Default: 'nndsvdar'
        Valid options::

            'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
                initialization (better for sparseness)
            'nndsvda': NNDSVD with zeros filled with the average of X
                (better when sparsity is not desired)
            'nndsvdar': NNDSVD with zeros filled with small random values
                (generally faster, less accurate alternative to NNDSVDa
                for when sparsity is not desired)
            int seed or RandomState: non-negative random matrices

    sparseness: 'data' | 'components' | None, default: None
        Where to enforce sparsity in the model.

    beta: double, default: 1
        Degree of sparseness, if sparseness is not None. Larger values mean
        more sparseness.

    eta: double, default: 0.1
        Degree of correctness to mantain, if sparsity is not None. Smaller
        values mean larger error.

    tol: double, default: 1e-4
        Tolerance value used in stopping conditions.

    max_iter: int, default: 200
        Number of iterations to compute.

    nls_max_iter: int, default: 2000
        Number of iterations in NLS subproblem.

    Attributes
    ----------
    `components_` : array, [n_components, n_features]
        Non-negative components of the data

    `reconstruction_err_` : number
        Frobenius norm of the matrix difference between the
        training data and the reconstructed data from the
        fit produced by the model. ``|| X - WH ||_2``
        Not computed for sparse input matrices because it is
        too expensive in terms of memory.

    Examples
    --------

    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import ProjectedGradientNMF
    >>> model = ProjectedGradientNMF(n_components=2, init=0)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ProjectedGradientNMF(beta=1, eta=0.1, init=0, max_iter=200, n_components=2,
                         nls_max_iter=2000, sparseness=None, tol=0.0001)
    >>> model.components_
    array([[ 0.77032744,  0.11118662],
           [ 0.38526873,  0.38228063]])
    >>> model.reconstruction_err_ #doctest: +ELLIPSIS
    0.00746...
    >>> model = ProjectedGradientNMF(n_components=2, init=0,
    ...                              sparseness='components')
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ProjectedGradientNMF(beta=1, eta=0.1, init=0, max_iter=200, n_components=2,
               nls_max_iter=2000, sparseness='components', tol=0.0001)
    >>> model.components_
    array([[ 1.67481991,  0.29614922],
           [-0.        ,  0.4681982 ]])
    >>> model.reconstruction_err_ #doctest: +ELLIPSIS
    0.513...

    Notes
    -----
    This implements

    C.-J. Lin. Projected gradient methods
    for non-negative matrix factorization. Neural
    Computation, 19(2007), 2756-2779.
    http://www.csie.ntu.edu.tw/~cjlin/nmf/

    P. Hoyer. Non-negative Matrix Factorization with
    Sparseness Constraints. Journal of Machine Learning
    Research 2004.

    NNDSVD is introduced in

    C. Boutsidis, E. Gallopoulos: SVD based
    initialization: A head start for nonnegative
    matrix factorization - Pattern Recognition, 2008
    http://www.cs.rpi.edu/~boutsc/files/nndsvd.pdf

    """

    def __init__(self, n_components=None, init="nndsvdar", sparseness=None,
                 beta=1, eta=0.1, tol=1e-4, max_iter=200, nls_max_iter=2000):
        BaseNMF.__init__(self, init=init, n_components=n_components)
        self.tol = tol
        if sparseness not in (None, 'data', 'components'):
            raise ValueError(
                'Invalid sparseness parameter: got %r instead of one of %r' %
                (sparseness, (None, 'data', 'components')))
        self.sparseness = sparseness
        self.beta = beta
        self.eta = eta
        self.max_iter = max_iter
        self.nls_max_iter = nls_max_iter

    def _update_W(self, X, H, W, tolW):
        n_samples, n_features = X.shape

        if self.sparseness == None:
            W, gradW, iterW = _nls_subproblem(X.T, H.T, W.T, tolW,
                                              self.nls_max_iter)
        elif self.sparseness == 'data':
            W, gradW, iterW = _nls_subproblem(
                    np.r_[X.T, np.zeros((1, n_samples))],
                    np.r_[H.T, np.sqrt(self.beta) *
                          np.ones((1, self.n_components))],
                    W.T, tolW, self.nls_max_iter)
        elif self.sparseness == 'components':
            W, gradW, iterW = _nls_subproblem(
                    np.r_[X.T, np.zeros((self.n_components, n_samples))],
                    np.r_[H.T, np.sqrt(self.eta) *
                          np.eye(self.n_components)],
                    W.T, tolW, self.nls_max_iter)

        return W, gradW, iterW

    def _update_H(self, X, H, W, tolH):
        n_samples, n_features = X.shape

        if self.sparseness == None:
            H, gradH, iterH = _nls_subproblem(X, W, H, tolH,
                                              self.nls_max_iter)
        elif self.sparseness == 'data':
            H, gradH, iterH = _nls_subproblem(
                    np.r_[X, np.zeros((self.n_components, n_features))],
                    np.r_[W, np.sqrt(self.eta) *
                          np.eye(self.n_components)],
                    H, tolH, self.nls_max_iter)
        elif self.sparseness == 'components':
            H, gradH, iterH = _nls_subproblem(
                    np.r_[X, np.zeros((1, n_features))],
                    np.r_[W, np.sqrt(self.beta) *
                          np.ones((1, self.n_components))],
                    H, tolH, self.nls_max_iter)

        return H, gradH, iterH

    def fit_transform(self, X, y=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data
        """
        X = atleast2d_or_csr(X)
        check_non_negative(X, "NMF.fit")

        n_samples, n_features = X.shape

        if not self.n_components:
            self.n_components = n_features

        W, H = self._init(X)

        gradW = (np.dot(W, np.dot(H, H.T))
                 - safe_sparse_dot(X, H.T, dense_output=True))
        gradH = (np.dot(np.dot(W.T, W), H)
                 - safe_sparse_dot(W.T, X, dense_output=True))
        init_grad = norm(np.r_[gradW, gradH.T])
        tolW = max(0.001, self.tol) * init_grad  # why max?
        tolH = tolW

        for n_iter in xrange(1, self.max_iter + 1):
            # stopping condition
            # as discussed in paper
            proj_norm = norm(np.r_[gradW[np.logical_or(gradW < 0, W > 0)],
                                   gradH[np.logical_or(gradH < 0, H > 0)]])
            if proj_norm < self.tol * init_grad:
                break

            # update W
            W, gradW, iterW = self._update_W(X, H, W, tolW)

            W = W.T
            gradW = gradW.T
            if iterW == 1:
                tolW = 0.1 * tolW

            # update H
            H, gradH, iterH = self._update_H(X, H, W, tolH)

            if iterH == 1:
                tolH = 0.1 * tolH

            self.comp_sparseness_ = _sparseness(H.ravel())
            self.data_sparseness_ = _sparseness(W.ravel())

            if not sp.issparse(X):
                self.reconstruction_err_ = norm(X - np.dot(W, H))

            self.components_ = H

        if n_iter == self.max_iter:
            warnings.warn("Iteration limit reached during fit")

        return W

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        """Transform the data X according to the fitted NMF model

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be transformed by the model

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data
        """
        X = atleast2d_or_csr(X)
        H = np.zeros((X.shape[0], self.n_components))
        for j in xrange(0, X.shape[0]):
            H[j, :], _ = nnls(self.components_.T, X[j, :])
        return H


class BetaNMF(BaseNMF):
    """Non negative factorization with beta divergence cost.

    Parameters
    ----------
    X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Data the model will be fit to.

    n_components: int or None
        Number of components, if n_components is not set all components
        are kept

    init:  'nndsvd' |  'nndsvda' | 'nndsvdar' | int | RandomState
        Method used to initialize the procedure.
        Default: 'nndsvdar'
        Valid options::

            'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
                initialization (better for sparseness)
            'nndsvda': NNDSVD with zeros filled with the average of X
                (better when sparsity is not desired)
            'nndsvdar': NNDSVD with zeros filled with small random values
                (generally faster, less accurate alternative to NNDSVDa
                for when sparsity is not desired)
            int seed or RandomState: non-negative random matrices

    update: 'gradient' | 'maxmin' | 'heuristic'
        Update methods::

            'gradient': simple alternate projected gradient descent
                on beta-divergence,
            'maxmin': Maximization-Minimization updates
            'heuristic': (default) Heuristic updates

    beta: double, default: 2
        Beta parameter of the divergence used to compute error between data
        and reconstruction.

    tol: double, default: 1e-4
        Tolerance value used in stopping conditions.

    max_iter: int, default: 200
        Number of iterations to compute.

    eta: double, default: 0.1
        Update coefficient. For gradient update only.

    subit: int, default: 10
        Number of sub-iterations to perform on W (resp. H) before switching
        to H (resp. W) update.

    Attributes
    ----------
    `components_` : array, [n_components, n_features]
        Non-negative components of the data

    Examples
    --------

    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import BetaNMF
    >>> model = BetaNMF(n_components=2, init=0)
    >>> model.fit(X) #doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    BetaNMF(beta=2, eps=1e-08, eta=0.1, init=0, max_iter=200, n_components=2,
        subit=10, tol=0.0001, update='heuristic')
    >>> model.components_
    array([[ 0.68495703,  0.36004651]
           [ 0.58376531,  0.04665704]])

    Notes
    -----
    This implements

    Févotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the β-divergence.  Neural Computation, 29(9),
    2421-2456. doi:http://dx.doi.org/10.1162/NECO_a_00168
    """

    updates = ('gradient', 'heuristic', 'maxmin')

    def __init__(self, n_components=None, beta=2, init=None,
            update='heuristic', tol=1e-4, max_iter=200, eps=1.e-8, subit=10,
            eta=0.1):
        BaseNMF.__init__(self, init=init, n_components=n_components)
        self.tol = tol
        self.beta = beta
        if update not in BetaNMF.updates:
            raise ValueError(
                'Invalid update parameter: got %r instead of one of %r' %
                (update, BetaNMF.updates))
        self.update = update
        self.max_iter = max_iter
        self.eps = eps
        # Only for max-min updates
        self.gamma_exp = self._gamma_exponent()
        # Only for gradient updates
        self.subit = subit
        self.eta = eta

    # Update rules

    def _update_W(self, X, W, H, weights=1.):
        if self.update == 'gradient':
            # Alternate projected gradient updates
            for _ in range(self.subit):
                gradW = self._grad_W(X, W, H, weights=weights)
                up_W = W - self.eta * gradW
                return up_W * (up_W > 0)
        elif self.update == 'maxmin':
            # Maximization-Minimization update
            return W * (self._heuristic_W(X, W, H, weights=weights)
                    ** self.gamma_exp)
        elif self.update == 'heuristic':
            # Heuristic update
            return W * self._heuristic_W(X, W, H, weights=weights)

    def _update_H(self, X, W, H, weights=1.):
        if self.update == 'gradient':
            # Alternate projected gradient updates
            for _ in range(self.subit):
                gradH = self._grad_H(X, W, H, weights=weights)
                up_H = H - self.eta * gradH
                return up_H * (up_H > 0)
        elif self.update == 'maxmin':
            # Maximization-Minimization update
            return H * (self._heuristic_H(X, W, H, weights=weights)
                    ** self.gamma_exp)
        elif self.update == 'heuristic':
            # Heuristic update
            return H * self._heuristic_H(X, W, H, weights=weights)

    def fit_transform(self, X, y=None, weights=1., _fit=True):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        weights: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Weights on the cost function used as coefficients on each
            element of the data. If smaller dimension is provided, standard
            numpy broadcasting is used.

        _fit: if True (default), update the model, else only compute transform

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data
        """
        X = atleast2d_or_csr(X)
        check_non_negative(X, "NMF.fit")

        n_samples, n_features = X.shape

        if not self.n_components:
            self.n_components = n_features

        W, H = self._init(X)

        if _fit:
            self.components_ = H

        prev_error = np.Inf
        tol = self.tol * n_samples * n_features

        for n_iter in xrange(1, self.max_iter + 1):
            # Stopping condition
            error = self.error(X, W, self.components_, weights=weights)
            if prev_error - error < tol:
                break
            prev_error = error

            # update W
            W = self._update_W(X, W, self.components_)

            if _fit:
                # update H
                self.components_ = self._update_H(X, W, self.components_)

        if n_iter == self.max_iter:
            warnings.warn("Iteration limit reached during fit")

        return W

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be decomposed

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X, **params):
        """Transform the data X according to the fitted NMF model

        Parameters
        ----------

        X: {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data matrix to be transformed by the model

        Returns
        -------
        data: array, [n_samples, n_components]
            Transformed data
        """
        params['_fit'] = False
        return self.fit_transform(X, **params)

    # Helpers for beta divergence and related updates

    def beta_divergence(self, X, Y, weights=1.):
        """Returns the self.beta-divergence from matrix X to Y.
        This is computed value-wise and then summed over all coefficients.
        """
        if self.beta == 0:
            q = X / (Y + self.eps)
            v = q - np.log(q) - 1.
        elif self.beta == 1:
            v = X * np.log(X / (Y + self.eps)) - X + Y
        else:
            v = X ** self.beta
            v += (self.beta - 1.) * ((Y + self.eps) ** self.beta)
            v -= self.beta * X * (Y + self.eps) ** (self.beta - 1.)
            v /= self.beta * (self.beta - 1.)
        return (weights * v).sum()

    def _gamma_exponent(self):
        """Exponent used for heuristic update from [Fevotte2011].
        """
        if self.beta < 1.:
            return 1. / (2. - self.beta)
        elif self.beta > 2:
            return 1. / (self.beta - 1)
        else:
            return 1.

    def _der_beta_div_der_y(self, X, Y):
        """Returns the result of the elemtent-wise application to matrices X
        and Y of the partial derivative of the self.beta-divergence, regarding
        second variable.
        """
        if self.beta == 0:
            # d_beta'(x|y) = 1 / y - x / y**2
            Yinv = 1. / (Y + self.eps)
            return Yinv - X * (Yinv ** 2)
        elif self.beta == 1:
            # d_beta'(x|y) = 1 - x / y
            return 1. - X / (Y + self.eps)
        else:
            # d_beta'(x|y) = y ** (beta - 1) - x * (y ** (beta - 2))
            return ((self.eps + Y) ** (self.beta - 1)
                    - X * ((Y + self.eps) ** (self.beta - 2)))

    def _grad_W(self, X, W, H, weights=1.):
        """Returns the gradient (in form of a matrix) of the loss function
        for given values of X, W, H.
        """
        return np.dot(weights * self._der_beta_div_der_y(X, np.dot(W, H)), H.T)

    def _grad_H(self, X, W, H, weights=1.):
        """Returns the gradient (in form of a matrix) of the loss function
        for given values of X, W, H.
        """
        return np.dot(W.T, weights * self._der_beta_div_der_y(X, np.dot(W, H)))
        # Could also be defined as grad_W(self.beta, X.T, H.T, W.T).T

    def _heuristic_W(self, X, W, H, weights=1.):
        reconstr = weights * np.dot(W, H)
        return (np.dot(X * ((reconstr + self.eps) ** (self.beta - 2)), H.T) /
                np.dot((reconstr + self.eps) ** (self.beta - 1), H.T))

    def _heuristic_H(self, X, W, H, weights=1.):
        reconstr = weights * np.dot(W, H)
        return (np.dot(W.T, X * ((reconstr + self.eps) ** (self.beta - 2))) /
                np.dot(W.T, (reconstr + self.eps) ** (self.beta - 1)))
        # Same remark than for gradient

    # Errors and performance estimations

    def error(self, X, W, H=None, weights=1.):
        if H is None:
            H = self.components_
        return self.beta_divergence(X, np.dot(W, H), weights=weights)

    # Projections

    def scale(self, W, H, factors):
        """Scale W columns and H rows inversely, according to the given
        coefficients.
        """
        factors = np.array(factors)[np.newaxis, :]
        s_W = W * (factors.T + self.eps)
        s_H = H / (factors.T + self.eps)
        return s_W, s_H


class NMF(ProjectedGradientNMF):
    __doc__ = ProjectedGradientNMF.__doc__
    pass
