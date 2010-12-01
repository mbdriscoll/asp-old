import unittest
import pylab as pl
import matplotlib as mpl
import itertools

from em import *

#input_data = np.recfromcsv('test.csv', names=None, dtype=np.float32)


class BasicTests(unittest.TestCase):
    def setUp(self):
        self.version_in = '1'
        N = 600
        self.M = 2
        np.random.seed(0)
        C = np.array([[0., -0.7], [3.5, .7]])
        Y = np.r_[
            np.dot(np.random.randn(N/2, 2), C),
            np.random.randn(N/2, 2) + np.array([3, 3]),
            ]
        self.X = np.array(Y, dtype=np.float32)
        self.gmm = GMM(self.M, self.version_in)

    def tearDown(self):
        self.plot(self.means, self.covars, self.X, "")

    def test_pure_python(self):
        self.means, self.covars = self.gmm.train_using_python(self.X)

    def test_generated(self):        
        self.means, self.covars = self.gmm.train(self.X)

    def plot(self, means, covars, X, name):
        splot = pl.subplot(111, aspect='equal', title=name)
        color_iter = itertools.cycle (['r', 'g', 'b', 'c'])
        pl.scatter(X.T[0], X.T[1], .8, color='k')
        for i, (mean, covar, color) in enumerate(zip(means, covars, color_iter)):
            v, w = np.linalg.eigh(covar)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan(u[1]/u[0])
            angle = 180 * angle / np.pi # convert to degrees
            ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)

        pl.show()

if __name__ == '__main__':
    unittest.main()
