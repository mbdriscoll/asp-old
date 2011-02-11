import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys
import math
import timeit

from em import *

def generate_synthetic_data(N):
    np.random.seed(0)
    C = np.array([[0., -0.7], [3.5, .7]])
    C1 = np.array([[-0.4, 1.7], [0.3, .7]])
    Y = np.r_[
        np.dot(np.random.randn(N/3, 2), C1),
        np.dot(np.random.randn(N/3, 2), C),
        np.random.randn(N/3, 2) + np.array([3, 3]),
        ]
    return Y.astype(np.float32)

class EMTester(object):

    def __init__(self, version_in, D, M, N):
        self.version_in = version_in
        self.M = M
        self.D = D
        self.N = N
        self.gmm = GMM(M, D)
        
        self.results = {}
        
        fromgen = generate_synthetic_data(self.N)
        fromfile = np.recfromcsv('IS1000a.csv', names=None, dtype=np.float32)

        self.X = fromfile

    def test_pure_python(self):
        means, covars = self.gmm.train_using_python(self.X)
        self.results['Pure'] = ('411', means, covars)

    def test_generated(self):        

        likelihood = self.gmm.train(self.X)
        means = self.gmm.components.means.reshape((self.gmm.M, self.gmm.D))
        covars = self.gmm.components.covars.reshape((self.gmm.M, self.gmm.D, self.gmm.D))
        self.results['ASP v'+self.version_in] = ('412', means, covars)
        return likelihood
        

    def test_train(self):
        self.test_pure_python()
        self.test_generated()


    def test_merge(self):
        self.merge_components()

    def test_ahc(self):
        self.test_pure_python()
        # try one train and one merge

        M_start = self.M
        M_end = 0
        plot_counter = 2
        

        for M in reversed(range(M_end, M_start)):

            print "======================== AHC loop: M = ", M, " ==========================="
            self.gmm.train(self.X)
        
            #plotting
            means = self.gmm.components.means.reshape((self.gmm.M, self.gmm.D)).copy()
            covars = self.gmm.components.covars.reshape((self.gmm.M, self.gmm.D, self.gmm.D)).copy()
            self.results['ASP v'+self.version_in+' M: '+str(M)] = ('41'+str(plot_counter), means, covars)
            plot_counter += 1

            #find closest components and merge
            if M > 0: #don't merge if there is only one component
                gmm_list = []
                count = 2
                for c1 in range(0, self.gmm.M):
                    for c2 in range(c1+1, self.gmm.M):
                        new_component, dist = self.gmm.compute_distance_rissanen(c1, c2)
                        gmm_list.append((dist, (c1, c2, new_component)))
                        print "gmm_list after append: ", gmm_list
                        
                #compute minimum distance
                min_c1, min_c2, min_component = min(gmm_list, key=lambda gmm: gmm[0])[1]
                self.gmm.merge_components(min_c1, min_c2, min_component)

    def time_ahc(self):
        M_start = self.M
        M_end = 0

        for M in reversed(range(M_end, M_start)):

            print "======================== AHC loop: M = ", M, " ==========================="
            self.gmm.train(self.X)

            #find closest components and merge
            if M > 0: #don't merge if there is only one component
                gmm_list = []
                count = 2
                for c1 in range(0, self.gmm.M):
                    for c2 in range(c1+1, self.gmm.M):
                        new_component, dist = self.gmm.compute_distance_rissanen(c1, c2)
                        gmm_list.append((dist, (c1, c2, new_component)))
                        
                #compute minimum distance
                min_c1, min_c2, min_component = min(gmm_list, key=lambda gmm: gmm[0])[1]
                self.gmm.merge_components(min_c1, min_c2, min_component)

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
    emt = EMTester(sys.argv[1], 19, 16, 158256)
    #emt.test_train()
    #t = timeit.Timer(emt.test_pure_python)
    t = timeit.Timer(emt.time_ahc)
    print t.timeit(number=1)
    #emt.gmm = GMM(3, 19, sys.argv[1])
    #print t.timeit(number=1)
    #emt.time_ahc()
 
