import unittest
import pylab as pl
import matplotlib as mpl
import itertools
import sys

from em import *



class EMTester(object):

    def __init__(self, version_in, M, D):
        self.version_in = version_in
        self.M = M
        self.D = D
        self.N = 600
        self.gmm = GMM(M, D, version_in)
        
        self.results = {}
        
        N = self.N
        np.random.seed(0)
        C = np.array([[0., -0.7], [3.5, .7]])
        C1 = np.array([[-0.4, 1.7], [0.3, .7]])
        Y = np.r_[
            np.dot(np.random.randn(N/2, 2), C1),
            np.dot(np.random.randn(N/2, 2), C),
            np.random.randn(N/2, 2) + np.array([3, 3]),
            ]
        self.X = np.array(Y, dtype=np.float32)
        #self.X = np.recfromcsv('test.csv', names=None, dtype=np.float32)

    def test_pure_python(self):
        means, covars = self.gmm.train_using_python(self.X)
        self.results['Pure'] = ('511', means, covars)

    def test_generated(self):        
        clusters = self.gmm.train(self.X)
        means = self.gmm.clusters.means.reshape((self.M, self.D))
        covars = self.gmm.clusters.covars.reshape((self.M, self.D, self.D))
        self.results['ASP v'+self.version_in] = ('512', means, covars)
        return means, covars
        
    # def merge_clusters(self):
    #     clusters = self.gmm.merge_2_closest_clusters()
    #     means = self.gmm.clusters.means
    #     covars = self.gmm.clusters.covars
    #     self.results['MERGE ASP v'+self.version_in] = ('313', means, covars)
    #     return means, covars
        
    def test_train(self):
        self.test_pure_python()
        self.test_generated()

    def test_merge(self):
        self.merge_clusters()

    def test_ahc(self):
        self.test_pure_python()
        # try one train and one merge
        self.test_generated()
        gmm_list = []
        count = 2
        for c1 in range(0, self.gmm.M):
            for c2 in range(c1+1, self.gmm.M):
                new_cluster, dist = self.gmm.compute_distance_riassen(c1, c2, self.gmm.D)
                gmm_list.append((dist, (c1, c2, new_cluster)))
                means = self.gmm.get_new_cluster_means(new_cluster, 1, self.gmm.D)
                covars = self.gmm.get_new_cluster_covars(new_cluster, 1, self.gmm.D)
                count+=1
                self.results['MERGE ASP '+str(c1)+'-'+str(c2)+" v:"+self.version_in] = ('51'+str(count), means, covars)
          
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
    emt = EMTester(sys.argv[1], 3, 2)
    #emt.test_train()
    emt.test_ahc()
    #emt.test_merge()
    emt.plot()
     
