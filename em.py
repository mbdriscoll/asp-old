import numpy as np
from numpy.random import *
from numpy import s_
import asp.codegen.templating.template as AspTemplate
import asp.jit.asp_module as asp_module
from codepy.cgen import *
from codepy.cuda import CudaModule
import pyublas
import sys
from imp import find_module
from os.path import join

class Clusters(object):
    
    def __init__(self, M, D, weights = None, means = None, covars = None):

        # self.weights = np.array([], dtype=np.float32)
        # self.means = np.array([], dtype=np.float32)
        # self.covars = np.array([], dtype=np.float32)
        
        self.weights = np.empty(M, dtype=np.float32)
        self.means = np.empty(M*D, dtype=np.float32)
        self.covars = np.empty(M*D*D, dtype=np.float32)

        # if weights is None:
        #     self.weights = self.init_random_weights(M)
        # else:
        #     self.weights = weights.copy(deep=True)    
            
        # if means is None:
        #     self.means = self.init_random_means(M, D)
        # else:
        #     self.means = means.copy(deep=True)
                    
        # if covars is None:
        #     self.covars = self.init_random_covars(M, D)
        # else:
        #     self.covars = covars.copy(deep=True)

    def init_random_weights(self, M):
        return numpy.random.random((M))

    def init_random_means(self, M, D):
        return numpy.random.random((M,D))

    def init_random_covars(self, M, D):
        return numpy.random.random((M, D, D))

    #this is inefficient - can we get away with no copy?
    def shrink_clusters(self, new_M, D):
        np.delete(self.weights, s_[new_M:])
        np.delete(self.means, s_[new_M*D:])
        np.delete(self.covars, s_[new_M*D*D:])
        return self.weights, self.means, self.covars
        
class GMM(object):

    #Singleton ASP mode shared by all instances of GMM
    asp_mod = None    
    def get_asp_mod(self): return GMM.asp_mod or self.initialize_asp_mod()

    variant_param_default = {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256'],
            'num_event_blocks': ['128'],
            'max_num_dimensions': ['50'],
            'max_num_clusters': ['128'],
            'diag_only': ['0'],
            'max_iters': ['10'],
            'min_iters': ['1'],
            'covar_version_name': ['1', '2A', '2B', '3']
    }

    # flags to keep track of memory allocation
    event_data_gpu_copy = None
    event_data_cpu_copy = None
    cluster_data_gpu_copy = None
    cluster_data_cpu_copy = None

    def internal_alloc_event_data(self, X):
        #TODO: test for not null
        #if not X.any(): return
        if not np.array_equal(self.event_data_gpu_copy, X):
            if GMM.event_data_gpu_copy is not None:
                self.internal_free_event_data()
            self.get_asp_mod().alloc_events_on_CPU(X, X.shape[0], X.shape[1])
            self.get_asp_mod().alloc_events_on_GPU(X.shape[0], X.shape[1])
            self.get_asp_mod().copy_event_data_CPU_to_GPU(X.shape[0], X.shape[1])
            GMM.event_data_gpu_copy = X
            GMM.event_data_cpu_copy = X

    def internal_free_event_data(self):
        if GMM.event_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_events_on_GPU()
            GMM.event_data_gpu_copy = None
        if self.event_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_events_on_CPU()
            GMM.event_data_cpu_copy = None

    def internal_alloc_cluster_data(self):

        #TODO: test for not null
        #if not self.clusters.weights.size: return
        if GMM.cluster_data_gpu_copy != self.clusters:
            if GMM.cluster_data_gpu_copy:
                self.internal_free_cluster_data()
            self.get_asp_mod().alloc_clusters_on_GPU(self.M, self.D)
            self.get_asp_mod().alloc_clusters_on_CPU(self.M, self.D, self.clusters.weights, self.clusters.means, self.clusters.covars)
            self.get_asp_mod().copy_cluster_data_CPU_to_GPU(self.M, self.D)
            GMM.cluster_data_gpu_copy = self.clusters
            GMM.cluster_data_cpu_copy = self.clusters
            
    def internal_free_cluster_data(self):
        if GMM.cluster_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_clusters_on_GPU()
            GMM.cluster_data_gpu_copy = None
        if GMM.cluster_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_clusters_on_CPU()
            GMM.cluster_data_cpu_copy = None


    def __init__(self, M, D, variant_param_space=None, means=None, covars=None, weights=None):
        self.M = M
        self.D = D
        self.variant_param_space = variant_param_space or GMM.variant_param_default

        self.clusters = Clusters(M, D, weights, means, covars)

    def initialize_asp_mod(self):

        # Create ASP module
        GMM.asp_mod = asp_module.ASPModule(use_cuda=True)
        cuda_mod = GMM.asp_mod.cuda_module

        #Add decls to preamble necessary for linking to compiled CUDA sources
        #TODO: Would it be better to pull in this preamble stuff from a file rather than have it  all sitting here?
        cluster_t_decl =""" 
            typedef struct clusters_struct {
                float* N;        // expected # of pixels in cluster: [M]
                float* pi;       // probability of cluster in GMM: [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the cluster: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } clusters_t;"""
        GMM.asp_mod.add_to_preamble(cluster_t_decl)

        # Render C/CUDA source templates based on inputs
        c_base_tpl = AspTemplate.Template(filename="templates/gaussian.mako")
        cu_base_tpl = AspTemplate.Template(filename="templates/theta_kernel_base.mako")
        c_main_tpl = AspTemplate.Template(filename="templates/train.mako")
        cu_kern_tpl = AspTemplate.Template(filename="templates/theta_kernel.mako")
        c_decl_tpl = AspTemplate.Template(filename="templates/gaussian_decl.mako") 

        import copy
        variant_param_dict = {}
        params_list = []
        c_main_rend_list = []
        c_decl_rend_list = []
        cu_kern_rend_list = []
        for k, vals in self.variant_param_space.iteritems():
            variant_param_dict[k] = vals[0]
        for v in self.variant_param_space['covar_version_name']:
            variant_param_dict['covar_version_name'] = v
            params_list.append(copy.deepcopy(variant_param_dict))

        list_of_suffixes = []
        for p in params_list:
            vals = p.values()
            c_main_rend_list.append(  c_main_tpl.render( param_val_list = vals, **p))
            cu_kern_rend_list.append(cu_kern_tpl.render( param_val_list = vals, **p))
            c_decl_rend_list.append(  c_decl_tpl.render( param_val_list = vals, **p))
            list_of_suffixes.append(vals)

        c_base_rend = c_base_tpl.render()
        cu_base_rend = cu_base_tpl.render()

        GMM.asp_mod.module.add_to_module([Line(c_base_rend)])
        cuda_mod.add_to_module([Line(cu_base_rend)])

        GMM.asp_mod.add_function_with_variants( c_main_rend_list, 
                                                "train", 
                                                [ 'train_'+'_'.join(vl) for vl in list_of_suffixes ], 
                                                lambda name, *args, **kwargs: (name, args[0], args[1], args[2]) )
        for rend in cu_kern_rend_list:
            cuda_mod.add_to_module([Line(rend)])
        for rend in c_decl_rend_list:
            GMM.asp_mod.add_to_preamble(rend)

        # Add set GPU device function
        GMM.asp_mod.add_function("", fname="set_GPU_device")
        
        # Add malloc and copy functions
        GMM.asp_mod.add_function("", fname="alloc_events_on_CPU")
        GMM.asp_mod.add_function("", fname="alloc_events_on_GPU")
        GMM.asp_mod.add_function("", fname="alloc_clusters_on_CPU")
        GMM.asp_mod.add_function("", fname="alloc_clusters_on_GPU")
        #GMM.asp_mod.add_function("", fname="alloc_temp_cluster_on_CPU")
        
        GMM.asp_mod.add_function("", fname="copy_event_data_CPU_to_GPU")
        GMM.asp_mod.add_function("", fname="copy_cluster_data_CPU_to_GPU")
        GMM.asp_mod.add_function("", fname="copy_cluster_data_GPU_to_CPU")

        # Add dealloc functions
        GMM.asp_mod.add_function("", fname="dealloc_events_on_CPU")
        GMM.asp_mod.add_function("", fname="dealloc_events_on_GPU")
        GMM.asp_mod.add_function("", fname="dealloc_clusters_on_CPU")
        GMM.asp_mod.add_function("", fname="dealloc_clusters_on_GPU")
        GMM.asp_mod.add_function("", fname="dealloc_temp_clusters_on_CPU")
        
        # Add getter functions
        GMM.asp_mod.add_function("", fname="get_temp_cluster_pi")
        GMM.asp_mod.add_function("", fname="get_temp_cluster_means")
        GMM.asp_mod.add_function("", fname="get_temp_cluster_covars")
        
        # Add merge clusters function
        GMM.asp_mod.add_function("", fname="relink_clusters_on_CPU")
        GMM.asp_mod.add_function("", fname="compute_distance_rissanen")
        GMM.asp_mod.add_function("", fname="merge_clusters")

        #Add necessary headers
        #TODO: Figure out whether we can free ourselves from cutils
        host_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h','pyublas/numpy.hpp' ]
        host_project_header_names = [ 'cutil_inline.h'] 
        for x in host_system_header_names: GMM.asp_mod.add_to_preamble([Include(x, True)])
        for x in host_project_header_names: GMM.asp_mod.add_to_preamble([Include(x, False)])

        GMM.asp_mod.add_to_init("""boost::python::class_<clusters_struct>("Clusters");
            boost::python::scope().attr("clusters") = boost::python::object(boost::python::ptr(&clusters));""")

        # GMM.asp_mod.add_to_init("""boost::python::class_<return_cluster_container>("ReturnClusterContainer")
        #    .def(pyublas::by_value_rw_member( "distance", &return_cluster_container::distance)) ;
        #     boost::python::scope().attr("cluster_distance") = boost::python::object(boost::python::ptr(&ret));""")

        GMM.asp_mod.add_to_init("""boost::python::class_<return_cluster_container>("ReturnClusterContainer")
            .def(pyublas::by_value_rw_member( "new_cluster", &return_cluster_container::cluster))
            .def(pyublas::by_value_rw_member( "distance", &return_cluster_container::distance)) ;
            boost::python::scope().attr("cluster_distance") = boost::python::object(boost::python::ptr(&ret));""")
        
        #Add headers, decls and rendered source to cuda_module
        cuda_mod.add_to_preamble([Include('stdio.h',True)])
        cuda_mod.add_to_preamble([Line(cluster_t_decl)])
        
        # Setup toolchain and compile
        def pyublas_inc():
            file, pathname, descr = find_module("pyublas")
            return join(pathname, "..", "include")
        def numpy_inc():
            file, pathname, descr = find_module("numpy")
            return join(pathname, "core", "include")

        nvcc_toolchain = GMM.asp_mod.nvcc_toolchain
        nvcc_toolchain.cflags += ["-arch=sm_20"]
        GMM.asp_mod.toolchain.add_library("project",['.','./include',pyublas_inc(),numpy_inc()],[],[])
        nvcc_toolchain.add_library("project",['.','./include'],[],[])

        #TODO: Get rid of awful hardcoded paths necessitaty by cutils
        GMM.asp_mod.toolchain.add_library("cutils",['/home/egonina/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/egonina/NVIDIA_GPU_Computing_SDK/C/lib','/home/egonina/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])
        nvcc_toolchain.add_library("cutils",['/home/egonina/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/egonina/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/egonina/NVIDIA_GPU_Computing_SDK/C/lib','/home/egonina/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])

        GMM.asp_mod.set_GPU_device(0);        
        
        #print GMM.asp_mod.module.generate()
        GMM.asp_mod.compile()

        return GMM.asp_mod

    def __del__(self):
        self.internal_free_event_data()
        self.internal_free_cluster_data()
    
    def train_using_python(self, input_data):
        from scikits.learn import mixture
        clf = mixture.GMM(n_states=self.M, cvtype='full')
        clf.fit(input_data)
        return clf.means, clf.covars
    
    def train(self, input_data):
        N = input_data.shape[0] #TODO: handle types other than np.array?
        D = input_data.shape[1]
        self.internal_alloc_event_data(input_data)
        self.internal_alloc_cluster_data()

        likelihood = self.get_asp_mod().train(self.M, D, N, input_data)
        return likelihood

    def merge_clusters(self, min_c1, min_c2, min_cluster):
        self.get_asp_mod().merge_clusters(min_c1, min_c2, min_cluster, self.M, self.D)
        self.M -= 1
        w, m, c = self.clusters.shrink_clusters(self.M, self.D)
        self.get_asp_mod().relink_clusters_on_CPU(w, m, c)
        self.get_asp_mod().dealloc_temp_clusters_on_CPU()
        return 

    def compute_distance_rissanen(self, c1, c2):
        self.get_asp_mod().compute_distance_rissanen(c1, c2, self.D)
        new_cluster = self.get_asp_mod().compiled_module.cluster_distance.new_cluster
        dist = self.get_asp_mod().compiled_module.cluster_distance.distance
        return new_cluster, dist

    def get_new_cluster_means(self, new_cluster):
        return self.get_asp_mod().get_temp_cluster_means(new_cluster, self.D).reshape((1, self.D))

    def get_new_cluster_covars(self, new_cluster):
        return self.get_asp_mod().get_temp_cluster_covars(new_cluster, self.D).reshape((1, self.D, self.D))
    

