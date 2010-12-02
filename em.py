import numpy as np
import asp.codegen.templating.template as AspTemplate
import asp.jit.asp_module as asp_module
from codepy.cgen import *
from codepy.cuda import CudaModule
import pyublas
import sys
from imp import find_module
from os.path import join

class GMM(object):
    def __init__(self, M, version_in):
        self.M = M

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
                float* memberships; // Fuzzy memberships: [N*M]
            } clusters_t;"""
        cuda_launch_decls ="""
            void seed_clusters_launch(float* d_fcs_data_by_event, clusters_t* d_clusters, int num_dimensions, int original_num_clusters, int num_events);
            void constants_kernel_launch(clusters_t* d_clusters, int original_num_clusters, int num_dimensions);
            void estep1_launch(float* d_fcs_data_by_dimension, clusters_t* d_clusters, int num_dimensions, int num_events, float* d_likelihoods, int num_clusters);
            void estep2_launch(float* d_fcs_data_by_dimension, clusters_t* d_clusters, int num_dimensions, int num_clusters, int num_events, float* d_likelihoods);
            void mstep_N_launch(float* d_fcs_data_by_event, clusters_t* d_clusters, int num_dimensions, int num_clusters, int num_events);
            void mstep_means_launch(float* d_fcs_data_by_dimension, clusters_t* d_clusters, int num_dimensions, int num_clusters, int num_events);
            void mstep_covar_launch_CODEVAR_1A(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
            void mstep_covar_launch_CODEVAR_2A(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
            void mstep_covar_launch_CODEVAR_2B(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
            void mstep_covar_launch_CODEVAR_3A(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
            """
        self.aspmod.add_to_preamble([Line(cluster_t_decl)])
        self.aspmod.add_to_preamble([Line(cuda_launch_decls)])


        #Add necessary headers
        #TODO: Figure out whether we can free ourselves from cutils
        host_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h','pyublas/numpy.hpp' ]
        host_project_header_names = [ 'cutil_inline.h'] 
        for x in host_system_header_names: self.aspmod.add_to_preamble([Include(x, True)])
        for x in host_project_header_names: self.aspmod.add_to_preamble([Include(x, False)])

        #Add Boost.Python registration of container class used to return data
        #TODO: Allow pointers to GPU data to be returned
        self.aspmod.add_to_init("""boost::python::class_<return_array_container>("ReturnArrayContainer")
            .def(pyublas::by_value_rw_member( "means", &return_array_container::means))
            .def(pyublas::by_value_rw_member( "covars", &return_array_container::covars)) ;
            boost::python::scope().attr("trained") = boost::python::object(boost::python::ptr(&ret));""")

        self.aspmod.add_to_init("""boost::python::class_<clusters_struct>("Clusters");
            boost::python::scope().attr("clusters") = boost::python::object(boost::python::ptr(&clusters));""")

        get_means_func = """ pyublas::numpy_array<float> get_means(clusters_t* c, int D, int M){
            pyublas::numpy_array<float> ret = pyublas::numpy_array<float>(D*M);
            std::copy( c->means, c->means+D*M, ret.begin());
            return ret;}"""

        self.aspmod.add_function(get_means_func, fname="get_means")

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
        self.aspmod.toolchain.add_library("cutils",['/home/henry/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/henry/NVIDIA_GPU_Computing_SDK/C/lib','/home/henry/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])
        nvcc_toolchain.add_library("cutils",['/home/henry/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/henry/NVIDIA_GPU_Computing_SDK/C/lib','/home/henry/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])

        self.aspmod.compile()


    def train_using_python(self, input_data):
        from scikits.learn import mixture
        clf = mixture.GMM(n_states=self.M, cvtype='full')
        clf.fit(input_data)
        return clf.means, clf.covars


    def train(self, input_data):
        N = input_data.shape[0] #TODO: handle types other than np.array?
        D = input_data.shape[1]
        #train( int device, int original_num_clusters, int num_dimensions, int num_events, pyublas::numpy_array<float> input_data )
        self.aspmod.train( 0, self.M, D, N, input_data)
        means = self.aspmod.compiled_module.trained.means.reshape((self.M, D))
        covars = self.aspmod.compiled_module.trained.covars.reshape((self.M, D, D))

        return means, covars
