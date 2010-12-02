import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys

from em import *



class EMTester(object):
    def __init__(self, version_in, M):
        self.version_in = version_in
        self.M = M
        self.D = 2
        self.N = 600
        self.gmm = GMM(M, version_in)

        self.results = {}

        N = self.N
        np.random.seed(0)
        C = np.array([[0., -0.7], [3.5, .7]])
        Y = np.r_[
            np.dot(np.random.randn(N/2, 2), C),
            np.random.randn(N/2, 2) + np.array([3, 3]),
            ]
        self.X = np.array(Y, dtype=np.float32)
        #self.X = np.recfromcsv('test.csv', names=None, dtype=np.float32)

    def test_pure_python(self):
        means, covars = self.gmm.train_using_python(self.X)
        self.results['Pure'] = ('211', means, covars)

    def test_generated(self):        
        means, covars = self.gmm.train(self.X)
        print self.gmm.get_asp_mod().get_means(self.gmm.get_asp_mod().compiled_module.clusters, self.D, self.M)
        self.results['ASP v'+self.version_in] = ('212', means, covars)

    def test(self):
        self.test_generated()
        self.test_pure_python()

    def plot(self):
        for t, r in self.results.iteritems():
            splot = pl.subplot(r[0], title=t)
            color_iter = itertools.cycle (['r', 'g', 'b', 'c'])
            pl.scatter(self.X.T[0], self.X.T[1], .8, color='k')
            for i, (mean, covar, color) in enumerate(zip(r[1], r[2], color_iter)):
                v, w = np.linalg.eigh(covar)
                u = w[0] / np.linalg.norm(w[0])
                angle = np.arctan(u[1]/u[0])
                angle = 180 * angle / np.pi
                ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                splot.add_artist(ell)
        pl.show()

if __name__ == '__main__':
    emt = EMTester(sys.argv[1], 2)
    emt.test()
    emt.plot()
