import unittest

import numpy as np

from ..nmf import _generalized_KL, KLdivNMF, _normalize_sum, _scale


def random_NN_matrix(h, w):
    return np.abs(np.random.random((h, w)))


def is_NN(a):
    return np.all(a >= 0)


class TestNormalizeSum(unittest.TestCase):

    def test_same_shape_on_1D(self):
        a = np.random.random((3,))
        norm = _normalize_sum(a, axis=0)
        self.assertTrue(np.alltrue(a.shape == norm.shape))
    
    def test_same_shape_on_2D(self):
        a = np.random.random((2, 4))
        norm = _normalize_sum(a, axis=np.random.randint(2))
        self.assertTrue(np.alltrue(a.shape == norm.shape))
    
    def test_same_shape_on_3D(self):
        a = np.random.random((1, 2, 3))
        norm = _normalize_sum(a, axis=np.random.randint(3))
        self.assertTrue(np.alltrue(a.shape == norm.shape))
    
    def test_correct_on_1D(self):
        a = np.random.random((5,))
        norm = _normalize_sum(a, axis=0)
        self.assertTrue(np.allclose(1., np.sum(norm)))

    def test_correct_on_2D_axis0(self):
        a = np.array([[0., 1., 3.], [2., 3., 3.]])
        norm = _normalize_sum(a, axis=0)
        ok = np.array([[0., .25, .5], [1., .75, .5]])
        self.assertTrue(np.alltrue(norm == ok))

    def test_correct_on_2D_axis1(self):
        a = np.array([[0., 1., 3.], [2., 3., 3.]])
        norm = _normalize_sum(a, axis=1)
        ok = np.array([[0., .25, .75], [.25, .375, .375]])
        self.assertTrue(np.alltrue(norm == ok))

    def test_correct_on_3D(self):
        a = np.random.random((2, 4, 5))
        ax = np.random.randint(3)
        norm = _normalize_sum(a, axis=ax)
        self.assertTrue(np.allclose(np.sum(norm, ax), 1.))

    def test_error_on_wrong_axis(self):
        a = np.random.random((2, 3, 4))
        with self.assertRaises(ValueError):
            _normalize_sum(a, axis=3)


class TestScale(unittest.TestCase):

    def test_shape(self):
        mtx = np.zeros((3, 4))
        fact = np.zeros((3,))
        self.assertEqual(mtx.shape, _scale(mtx, fact, axis=1).shape)

    def test_error_on_wrong_axis(self):
        mtx = np.zeros((3, 4))
        fact = np.zeros((4,))
        with self.assertRaises(ValueError):
            _scale(mtx, fact, axis=3)

    def test_error_on_3D_array(self):
        mtx = np.zeros((3, 4, 6))
        fact = np.zeros((3,))
        with self.assertRaises(ValueError):
            _scale(mtx, fact, axis=1)

    def test_error_on_1D_array(self):
        mtx = np.zeros((3,))
        fact = np.zeros((3,))
        with self.assertRaises(ValueError):
            _scale(mtx, fact, axis=1)

    def test_error_on_wrong_factor_shape(self):
        mtx = np.zeros((3, 4))
        fact = np.zeros((2,))
        with self.assertRaises(ValueError):
            _scale(mtx, fact, axis=1)

    def test_scale_lines(self):
        mtx = np.array([[1, 2, 3], [4, 5, 6]])
        fact = np.array([2, 3])
        scaled = _scale(mtx, fact, axis=1)
        ok = np.array([[2, 4, 6], [12, 15, 18]])
        self.assertTrue(np.allclose(ok, scaled))

    def test_scale_columns(self):
        mtx = np.array([[1, 2, 3], [4, 5, 6]])
        fact = np.array([3, 2, 1])
        scaled = _scale(mtx, fact, axis=0)
        ok = np.array([[3, 4, 3], [12, 10, 6]])
        self.assertTrue(np.allclose(ok, scaled))


class TestGenenralizedKL(unittest.TestCase):

    def setUp(self):
        self.x = random_NN_matrix(10, 15)
        self.y = random_NN_matrix(10, 15)

    def test_returns_scalar(self):
        self.assertTrue(np.isscalar(_generalized_KL(self.x, self.y)))

    def test_raises_ValueError_0(self):
        with self.assertRaises(ValueError):
            _generalized_KL(self.x[:-1, :], self.y)

    def test_raises_ValueError_1(self):
        with self.assertRaises(ValueError):
            _generalized_KL(self.x[:, 1:], self.y)

    def test_is_NN(self):
        self.assertTrue(_generalized_KL(self.x, self.y) >= 0)

    def test_is_0_on_same(self):
        self.assertTrue(np.allclose(_generalized_KL(self.x, self.x), 0))

    def test_is_1_homogenous(self):
        dkl = _generalized_KL(self.x, self.y)
        a = np.random.random()
        adkl = _generalized_KL(a * self.x, a * self.y)
        self.assertTrue(np.allclose(a * dkl, adkl))

    def test_values(self):
        x = np.zeros((4, 2))
        x[1, 1] = 1
        y = .5 * np.ones((4, 2))
        dkl = _generalized_KL(x, y)
        ok = np.log(2.) + 3.
        print dkl, ok
        self.assertTrue(np.allclose(dkl, ok))


class TestUpdates(unittest.TestCase):

    n_samples = 20
    n_components = 3
    n_features = 30

    def setUp(self):
        self.X = random_NN_matrix(self.n_samples, self.n_features)
        self.W = random_NN_matrix(self.n_samples, self.n_components)
        self.H = random_NN_matrix(self.n_components, self.n_features)
        self.nmf = KLdivNMF(n_components=3, init=None, tol=1e-4,
            max_iter=200, eps=1.e-8, subit=10)
        self.nmf.components_ = self.H

    def test_W_remains_NN(self):
        W = self.nmf._updated_W(self.X, self.W, self.H)
        self.assertTrue(is_NN(W))

    def test_H_remains_NN(self):
        H = self.nmf._updated_H(self.X, self.W, self.H)
        self.assertTrue(is_NN(H))

    def test_decreases_KL(self):
        dkl_prev = _generalized_KL(self.X, self.W.dot(self.H))
        W = self.nmf._update(self.X, self.W, _fit=True)
        dkl_next = _generalized_KL(self.X, W.dot(self.nmf.components_))
        self.assertTrue(dkl_prev > dkl_next)

    def test_no_compenents_update(self):
        self.nmf._update(self.X, self.W, _fit=False)
        self.assertTrue((self.nmf.components_ == self.H).all())


class TestFitTransform(unittest.TestCase):

    def setUp(self):
        self.nmf = KLdivNMF(n_components=3, init=None, tol=1e-4,
            max_iter=200, eps=1.e-8, subit=10)

    def test_cv(self):
        X = random_NN_matrix(10, 5)
        W, errors = self.nmf.fit_transform(X, return_errors=True)
        # Last errors should be very close
        self.assertTrue(abs(errors[-1] - errors[-2]) < errors[0] * 1.e-2)

    def test_zero_error_on_fact_data(self):
        X = np.dot(random_NN_matrix(5, 2), random_NN_matrix(2, 3))
        W, errors = self.nmf.fit_transform(X, return_errors=True)
        print errors[-1] / errors[0]
        self.assertTrue(errors[-1] < errors[0] * 1.e-3)
