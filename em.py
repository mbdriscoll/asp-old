import numpy as np
from numpy.random import *
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


class GMM(object):
    
    # flags to keep track of memory allocation
    event_data_gpu_copy = None
    cluster_data_gpu_copy = None

    def internal_alloc_event_data(self, X):
        #TODO: test for not null
        #if not X.any(): return
        if self.event_data_gpu_copy != X:
            if self.event_data_gpu_copy:
                self.internal_free_event_data()
            self.aspmod.alloc_events_on_CPU(X, X.shape[0], X.shape[1])
            self.aspmod.alloc_events_on_GPU(X.shape[0], X.shape[1])
            self.aspmod.copy_event_data_CPU_to_GPU(X.shape[0], X.shape[1])
            self.event_data_gpu_copy = X

    def internal_free_event_data(self):
        #if self.event_data_gpu_copy:
            self.aspmod.dealloc_events_on_CPU()
            self.aspmod.dealloc_events_on_GPU()
            self.event_data_gpu_copy = None

    def internal_alloc_cluster_data(self):

        #TODO: test for not null
        #if not self.clusters.weights.size: return

        if self.cluster_data_gpu_copy != self.clusters:
            if self.cluster_data_gpu_copy:
                self.internal_free_cluster_data()
            self.aspmod.alloc_clusters_on_CPU(self.M, self.D, self.clusters.weights, self.clusters.means, self.clusters.covars)
            self.aspmod.alloc_clusters_on_GPU(self.M, self.D)
            self.aspmod.copy_cluster_data_CPU_to_GPU(self.M, self.D)
            self.cluster_data_gpu_copy = self.clusters
            
    def internal_free_cluster_data(self):
        #if self.cluster_data_gpu_copy:
            self.aspmod.dealloc_clusters_on_CPU()
            self.aspmod.dealloc_clusters_on_GPU()
            self.cluster_data_gpu_copy = None

    def __init__(self, M, D, version_in, means=None, covars=None, weights=None):
        self.M = M
        self.D = D

        self.clusters = Clusters(M, D, weights, means, covars)
            
        version_suffix_list = ['CODEVAR_1A', 'CODEVAR_2A', 'CODEVAR_2B', 'CODEVAR_3A']
        version_suffix_mapping = {'1' : 'CODEVAR_1A', 
                                  '2' : 'CODEVAR_2A',
                                  '2A': 'CODEVAR_2A',
                                  '2B': 'CODEVAR_2B',
                                  '3' : 'CODEVAR_3A',
                                  '3A': 'CODEVAR_3A' }
        version_suffix = version_suffix_mapping[version_in]

        # Render C/CUDA source templates based on inputs
        #TODO: Render "all possible" variants instead of picking a single one based on constructor parameter
        c_main_tpl = AspTemplate.Template(filename="templates/gaussian.mako")
        cu_kern_tpl = AspTemplate.Template(filename="templates/theta_kernel.mako")
        c_main_rend = c_main_tpl.render(
            num_blocks_estep = 16,
            num_threads_estep = 512,
            num_threads_mstep = 256,
            num_event_blocks = 128,
            max_num_dimensions = 50,
            max_num_clusters = 128,
            device_id = 0,
            diag_only = 0,
            max_iters = 10,
            min_iters = 10,
            enable_2b_buffer = 1 if version_suffix == 'CODEVAR_2B' else 0,
            version_suffix=version_suffix
            )
        cu_kern_rend = cu_kern_tpl.render(
            num_blocks_estep = 16,
            num_threads_estep = 512,
            num_threads_mstep = 256,
            num_event_blocks = 128,
            max_num_dimensions = 50,
            max_num_clusters = 128,
            device_id = 0,
            diag_only = 0,
            max_iters = 10,
            min_iters = 10,
            )

        # Create ASP module
        #TODO: Handle ASP compilation separately from GMM initialization.
        #TODO: Have one ASP module for all GMM instances
        self.aspmod = asp_module.ASPModule(use_cuda=True)

        # Add train function and all helpers, was main() of original code 
        #TODO: Add all rendered variants using add_function_with_variants()
        #TODO: Change variant selection to key off of parameter values as well as fname
        self.aspmod.add_function(c_main_rend, fname="train")

        # Add set GPU device function
        self.aspmod.add_function("", fname="set_GPU_device")
        
        # Add malloc and copy functions
        self.aspmod.add_function("", fname="alloc_events_on_CPU")
        self.aspmod.add_function("", fname="alloc_events_on_GPU")
        self.aspmod.add_function("", fname="alloc_clusters_on_CPU")
        self.aspmod.add_function("", fname="alloc_clusters_on_GPU")
        #self.aspmod.add_function("", fname="alloc_temp_cluster_on_CPU")
        
        self.aspmod.add_function("", fname="copy_event_data_CPU_to_GPU")
        self.aspmod.add_function("", fname="copy_cluster_data_CPU_to_GPU")
        self.aspmod.add_function("", fname="copy_cluster_data_GPU_to_CPU")

        # Add dealloc functions
        self.aspmod.add_function("", fname="dealloc_events_on_CPU")
        self.aspmod.add_function("", fname="dealloc_events_on_GPU")
        self.aspmod.add_function("", fname="dealloc_clusters_on_CPU")
        self.aspmod.add_function("", fname="dealloc_clusters_on_GPU")
        #self.aspmod.add_function("", fname="dealloc_temp_cluster_on_CPU")
        
        # Add getter functions
        self.aspmod.add_function("", fname="get_temp_cluster_pi")
        self.aspmod.add_function("", fname="get_temp_cluster_means")
        self.aspmod.add_function("", fname="get_temp_cluster_covars")
        
        # Add merge clusters function
        self.aspmod.add_function("", fname="compute_distance_riassen")
        self.aspmod.add_function("", fname="merge_2_closest_clusters")
        


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

        cuda_launch_decls ="""
            void seed_clusters_launch(float* d_fcs_data_by_event, clusters_t* d_clusters, int num_dimensions, int original_num_clusters, int num_events);
            void constants_kernel_launch(clusters_t* d_clusters, int original_num_clusters, int num_dimensions);
            void estep1_launch(float* d_fcs_data_by_dimension, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_events, float* d_likelihoods, int num_clusters);
            void estep2_launch(float* d_fcs_data_by_dimension, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events, float* d_likelihoods);
            void mstep_N_launch(float* d_fcs_data_by_event, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events);
            void mstep_means_launch(float* d_fcs_data_by_dimension, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events);
            void mstep_covar_launch_CODEVAR_1A(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
            void mstep_covar_launch_CODEVAR_2A(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
            void mstep_covar_launch_CODEVAR_2B(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
            void mstep_covar_launch_CODEVAR_3A(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
            """
        self.aspmod.add_to_preamble([Line(cluster_t_decl)])
        self.aspmod.add_to_preamble([Line(cuda_launch_decls)])

        #Add necessary headers
        #TODO: Figure out whether we can free ourselves from cutils
        host_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h','pyublas/numpy.hpp' ]
        host_project_header_names = [ 'cutil_inline.h'] 
        for x in host_system_header_names: self.aspmod.add_to_preamble([Include(x, True)])
        for x in host_project_header_names: self.aspmod.add_to_preamble([Include(x, False)])

        self.aspmod.add_to_init("""boost::python::class_<clusters_struct>("Clusters");
            boost::python::scope().attr("clusters") = boost::python::object(boost::python::ptr(&clusters));""")

        # self.aspmod.add_to_init("""boost::python::class_<return_cluster_container>("ReturnClusterContainer")
        #    .def(pyublas::by_value_rw_member( "distance", &return_cluster_container::distance)) ;
        #     boost::python::scope().attr("cluster_distance") = boost::python::object(boost::python::ptr(&ret));""")

        self.aspmod.add_to_init("""boost::python::class_<return_cluster_container>("ReturnClusterContainer")
            .def(pyublas::by_value_rw_member( "new_cluster", &return_cluster_container::cluster))
            .def(pyublas::by_value_rw_member( "distance", &return_cluster_container::distance)) ;
            boost::python::scope().attr("cluster_distance") = boost::python::object(boost::python::ptr(&ret));""")


        
        # Create cuda-device module
        cuda_mod = self.aspmod.cuda_module

        #Add headers, decls and rendered source to cuda_module
        cuda_mod.add_to_preamble([Include('stdio.h',True)])
        cuda_mod.add_to_preamble([Line(cluster_t_decl)])
        cuda_mod.add_to_module([Line(cu_kern_rend)])
        
        # Setup toolchain and compile
        def pyublas_inc():
            file, pathname, descr = find_module("pyublas")
            return join(pathname, "..", "include")
        def numpy_inc():
            file, pathname, descr = find_module("numpy")
            return join(pathname, "core", "include")
        nvcc_toolchain = self.aspmod.nvcc_toolchain
        nvcc_toolchain.cflags += ["-arch=sm_20"]
        self.aspmod.toolchain.add_library("project",['.','./include',pyublas_inc(),numpy_inc()],[],[])
        nvcc_toolchain.add_library("project",['.','./include'],[],[])

        #TODO: Get rid of awful hardcoded paths necessitaty by cutils
        self.aspmod.toolchain.add_library("cutils",['/home/egonina/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/egonina/NVIDIA_GPU_Computing_SDK/C/lib','/home/egonina/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])
        nvcc_toolchain.add_library("cutils",['/home/egonina/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/egonina/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/egonina/NVIDIA_GPU_Computing_SDK/C/lib','/home/egonina/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])

        #print self.aspmod.module.generate()
        self.aspmod.compile()

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

        self.aspmod.set_GPU_device(0);
        self.internal_alloc_event_data(input_data)
        self.internal_alloc_cluster_data()

        self.aspmod.train(self.M, D, N, input_data)
        return 

    def merge_2_closest_clusters(self):
        self.aspmod.merge_2_closest_clusters(self.clusters, self.M, self.D)
        self.M -= 1
        return 

    def compute_distance_riassen(self, c1, c2, D):
        self.aspmod.compute_distance_riassen(c1, c2, D)
        new_cluster = self.aspmod.compiled_module.cluster_distance.new_cluster
        dist = self.aspmod.compiled_module.cluster_distance.distance
        return new_cluster, dist

    def get_new_cluster_means(self, new_cluster, M, D):
        return self.aspmod.get_temp_cluster_means(new_cluster, D).reshape((M, D))

    def get_new_cluster_covars(self, new_cluster, M, D):
        return self.aspmod.get_temp_cluster_covars(new_cluster, D).reshape((M, D, D))
    


