import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys
import math
import timeit
import copy

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

    def __init__(self, from_file, variant_param_space, num_subps):
        self.results = {}
        self.variant_param_space = variant_param_space
        self.num_subplots = num_subps
        self.plot_id = num_subps/2*100 + 21
        if from_file:
            self.X = np.recfromcsv('IS1000a.csv', names=None, dtype=np.float32)
            self.D = self.X.shape[0]
            self.N = self.X.shape[1]
        else:
            self.D = 2
            self.N = 600
            self.X = generate_synthetic_data(self.N)

    def new_gmm(self, M):
        self.M = M
        self.gmm = GMM(self.M, self.D, self.variant_param_space)

    def test_cytosis_ahc(self):
        M_start = self.M
        M_end = 0
        plot_counter = 2
        
        for M in reversed(range(M_end, M_start)):

            print "======================== AHC loop: M = ", M+1, " ==========================="
            self.gmm.train(self.X)
        
            #plotting
            means = self.gmm.components.means.reshape((self.gmm.M, self.gmm.D))
            covars = self.gmm.components.covars.reshape((self.gmm.M, self.gmm.D, self.gmm.D))
            Y = self.gmm.predict(self.X)
            if(self.plot_id % 10 <= self.num_subplots):
                self.results['_'.join(['ASP v',str(self.plot_id-(100*self.num_subplots+11)),'@',str(self.gmm.D),str(self.gmm.M),str(self.N)])] = (str(self.plot_id), copy.deepcopy(means), copy.deepcopy(covars), copy.deepcopy(Y))
                self.plot_id += 1

            #find closest components and merge
            if M > 0: #don't merge if there is only one component
                gmm_list = []
                for c1 in range(0, self.gmm.M):
                    for c2 in range(c1+1, self.gmm.M):
                        new_component, dist = self.gmm.compute_distance_rissanen(c1, c2)
                        gmm_list.append((dist, (c1, c2, new_component)))
                        #print "gmm_list after append: ", gmm_list
                        
                #compute minimum distance
                min_c1, min_c2, min_component = min(gmm_list, key=lambda gmm: gmm[0])[1]
                self.gmm.merge_components(min_c1, min_c2, min_component)

    def time_cytosis_ahc(self):
        M_start = self.M
        M_end = 0

        for M in reversed(range(M_end, M_start)):

            print "======================== AHC loop: M = ", M+1, " ==========================="
            self.gmm.train(self.X)

            #find closest components and merge
            if M > 0: #don't merge if there is only one component
                gmm_list = []
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
            Y_ = r[3]
            for i, (mean, covar, color) in enumerate(zip(r[1], r[2], color_iter)):
                v, w = np.linalg.eigh(covar)
                u = w[0] / np.linalg.norm(w[0])
                pl.scatter(self.X.T[0,Y_==i], self.X.T[1,Y_==i], .8, color=color)
                angle = np.arctan(u[1]/u[0])
                angle = 180 * angle / np.pi
                ell = mpl.patches.Ellipse (mean, v[0], v[1], 180 + angle, color=color)
                ell.set_clip_box(splot.bbox)
                ell.set_alpha(0.5)
                splot.add_artist(ell)
        pl.show()
        
if __name__ == '__main__':
    num_subplots = 6
    variant_param_space = {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256'],
            'num_event_blocks': ['128'],
            'max_num_dimensions': ['50'],
            'max_num_components': ['128'],
            'diag_only': ['0'],
            'max_iters': ['10'],
            'min_iters': ['1'],
            'covar_version_name': ['V2A']
    }
    emt = EMTester(False, variant_param_space, num_subplots)
    emt.new_gmm(6)
    #t = timeit.Timer(emt.time_ahc)
    #print t.timeit(number=1)
    emt.test_cytosis_ahc()
    emt.plot()
 
