import pylab as pl
import matplotlib as mpl
import itertools
import sys
import pickle
import timeit
from em import *

class EMAutotuner(object):

    def __init__(self, variant_param_space, input_param_space):
        fromfile = np.recfromcsv('IS1000a.csv', names=None, dtype=np.float32)
        self.orig_X = fromfile
        self.input_list = [] #tuple (M, X)
        self.shaped_Xs = {}
        self.variant_param_space = variant_param_space
        self.variant_param_space_size = sum([len(v) for v in self.variant_param_space.values()])
        self.input_param_space = input_param_space
        self.generate_shaped_inputs(input_param_space.keys(), input_param_space.values(), {})

    def add_to_input_list(self, param_dict):
        D = int(param_dict['D'])
        N = int(param_dict['N'])
        M = int(param_dict['M'])
        new_X = self.shaped_Xs.setdefault((D,N), np.resize(self.orig_X,(N,D)))
        self.input_list.append((M, D, N, new_X))

    def generate_shaped_inputs(self, key_arr, val_arr_arr, current):
        idx = len(current)
        name = key_arr[idx]
        for v in val_arr_arr[idx]:
            current[name]  = v
            if idx == len(key_arr)-1:
                self.add_to_input_list(current)
            else:
                self.generate_shaped_inputs(key_arr, val_arr_arr, current)
        del current[name]

    def test_point(self, input_tuple):
        self.gmm = GMM(input_tuple[0], input_tuple[1], self.variant_param_space)
        likelihood = self.gmm.train(input_tuple[3])

    def search_space(self):
        for i in self.input_list:
            print "M=%d, D=%d, N=%d" % i[:3]
            for j in range(0, self.variant_param_space_size):
                self.test_point(i)
            self.gmm.asp_mod.save_func_variant_timings("train")    

if __name__ == '__main__':
    variant_param_space = {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256','512'],
            'num_event_blocks': ['32','128'],
            'max_num_dimensions': ['50'],
            'max_num_clusters': ['128'],
            'diag_only': ['0'],
            'max_iters': ['10'],
            'min_iters': ['10'],
            'covar_version_name': ['V1', 'V2A', 'V2B', 'V3']
    }
    input_param_space =  {
            'D': np.arange(2, 40, 6),
            'N': np.arange(10000, 100001, 20000),
            'M': np.arange(1, 101, 20)
    }
    emt = EMAutotuner(variant_param_space, input_param_space)
    emt.search_space()
 
