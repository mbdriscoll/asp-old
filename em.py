import numpy as np
import numpy.linalg as la
import asp.codegen.templating.template as AspTemplate
import asp.jit.asp_module as asp_module
from codepy.cgen import *
from codepy.cuda import CudaModule
from codepy.cgen.cuda import CudaGlobal
import pyublas
from imp import find_module
import sys
from os.path import join

class EM(object):
    def __init__(self, X, N, M, D):
        self.X = X
        self.N = N
        self.M = M
        self.D = D
    
    def train_using_python(self):
        from scikits.learn import mixture
        clf = mixture.GMM(n_states=self.M, cvtype='full')
        clf.fit(self.X)
        return clf.means, clf.covars


    def train_using_asp(self, version_in):

        version_suffix_list = ['CODEVAR_1A', 'CODEVAR_2A', 'CODEVAR_2B', 'CODEVAR_3A']
        version_suffix_mapping = {'1' : 'CODEVAR_1A', 
                                  '2' : 'CODEVAR_2A',
                                  '2A': 'CODEVAR_2A',
                                  '2B': 'CODEVAR_2B',
                                  '3' : 'CODEVAR_3A',
                                  '3A': 'CODEVAR_3A' }
        version_suffix = version_suffix_mapping[version_in]
        num_event_blocks = 128

        # Render C/CUDA source templates based on inputs

        c_main_tpl = AspTemplate.Template(filename="templates/gaussian.mako")
        cu_kern_tpl = AspTemplate.Template(filename="templates/theta_kernel.mako")
        c_main_rend = c_main_tpl.render(
            num_blocks_estep = 16,
            num_threads_estep = 512,
            num_threads_mstep = 256,
            num_event_blocks = num_event_blocks,
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
            num_event_blocks = num_event_blocks,
            max_num_dimensions = 50,
            max_num_clusters = 128,
            device_id = 0,
            diag_only = 0,
            max_iters = 10,
            min_iters = 10,
            )

        cluster_t_decl = """typedef struct 
        {
            // Key for array lengths
            //  N = number of events
            //  M = number of clusters
            //  D = number of dimensions
            float* N;        // expected # of pixels in cluster: [M]
            float* pi;       // probability of cluster in GMM: [M]
            float* constant; // Normalizing constant [M]
            float* avgvar;    // average variance [M]
            float* means;   // Spectral mean for the cluster: [M*D]
            float* R;      // Covariance matrix: [M*D*D]
            float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            float* memberships; // Fuzzy memberships: [N*M]
        } clusters_t;"""

        # Create host-side module

        aspmod = asp_module.ASPModule(use_cuda=True)

        host_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h','pyublas/numpy.hpp' ]
        host_project_header_names = [ 'cutil_inline.h'] 

        aspmod.add_to_init("""boost::python::class_<return_array_container>("ReturnArrayContainer")
            .def(pyublas::by_value_rw_member( "means", &return_array_container::means))
            .def(pyublas::by_value_rw_member( "covars", &return_array_container::covars)) ;
            boost::python::scope().attr("trained") = boost::python::object(boost::python::ptr(&ret));""")
        for x in host_system_header_names: aspmod.add_to_preamble([Include(x, True)])
        for x in host_project_header_names: aspmod.add_to_preamble([Include(x, False)])
        aspmod.add_to_preamble([Line(cluster_t_decl)])
        aspmod.add_function(c_main_rend, fname="train")
        #aspmod.module.mod_body.append(Line(open('invert_matrix.cpp', 'r').read()))

        # Create cuda-device module

        cuda_mod = aspmod.cuda_module
        cuda_mod.add_to_preamble([Include('stdio.h',True)])
        cuda_mod.add_to_preamble([Line(cluster_t_decl)])
        cuda_mod.add_to_module([Line(cu_kern_rend)])

        seed_clusters_statements = [ 'seed_clusters<<< 1, NUM_THREADS_MSTEP >>>( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, num_events);' ]
        constants_kernel_statements = [ 'constants_kernel<<<original_num_clusters, 64>>>(d_clusters,original_num_clusters,num_dimensions);' ]
        estep1_statements = [ 'estep1<<<dim3(NUM_BLOCKS_ESTEP,num_clusters), NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods);' ]
        estep2_statements = [ 'estep2<<<NUM_BLOCKS_ESTEP, NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);' ]
        mstep_N_statements = [ 'mstep_N<<<num_clusters, NUM_THREADS_MSTEP>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);' ]
        mstep_means_statements = ['dim3 gridDim1(num_clusters,num_dimensions);',
              'mstep_means<<<gridDim1, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);' ]
        mstep_covar_launch_versions = {
            'CODEVAR_1A':  [ 'dim3 gridDim2(num_clusters,num_dimensions*(num_dimensions+1)/2);',
                    'mstep_covariance<<<gridDim2, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);' ],
            'CODEVAR_2A': [ 'int num_blocks = num_clusters;',
                    'int num_threads = num_dimensions*(num_dimensions+1)/2;',
                    'mstep_covariance_2a<<<num_clusters, num_threads>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);' ],
            'CODEVAR_2B': [ 'int num_event_blocks = NUM_EVENT_BLOCKS;',
                    'int event_block_size = num_events%NUM_EVENT_BLOCKS == 0 ? num_events/NUM_EVENT_BLOCKS:num_events/(NUM_EVENT_BLOCKS-1);',
                    'dim3 gridDim2(num_clusters,num_event_blocks);',
                    'int num_threads = num_dimensions*(num_dimensions+1)/2;',
                    'mstep_covariance_2b<<<gridDim2, num_threads>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events, event_block_size, num_event_blocks, temp_buffer_2b);' ],
            'CODEVAR_3A': [ 'int num_blocks = num_dimensions*(num_dimensions+1)/2;',
                    'mstep_covariance_3a<<<num_blocks, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);' ] }

        seed_clusters_launch_func = FunctionBody(  
                        FunctionDeclaration(Value('void', 'seed_clusters_launch'),
                                        [   Value('float*', 'd_fcs_data_by_event'), 
                                            Value('clusters_t*', 'd_clusters'), 
                                            Value('int', 'num_dimensions'), 
                                            Value('int', 'original_num_clusters'), 
                                            Value('int', 'num_events')  ]),
                        Block([Statement(s) for s in seed_clusters_statements]) )
        constants_kernel_launch_func = FunctionBody(  
                        FunctionDeclaration(Value('void', 'constants_kernel_launch'),
                                        [   Value('clusters_t*', 'd_clusters'), 
                                            Value('int', 'original_num_clusters'), 
                                            Value('int', 'num_dimensions') ]),
                        Block([Statement(s) for s in constants_kernel_statements]) )
        estep1_launch_func = FunctionBody(
                        FunctionDeclaration(Value('void', 'estep1_launch'),
                                        [   Value('float*', 'd_fcs_data_by_dimension'), 
                                            Value('clusters_t*', 'd_clusters'), 
                                            Value('int', 'num_dimensions'), 
                                            Value('int', 'num_events'),
                                            Value('float*', 'd_likelihoods'),
                                            Value('int', 'num_clusters') ]),
                        Block([Statement(s) for s in estep1_statements]) )
        estep2_launch_func = FunctionBody(
                        FunctionDeclaration(Value('void', 'estep2_launch'),
                                        [   Value('float*', 'd_fcs_data_by_dimension'), 
                                            Value('clusters_t*', 'd_clusters'), 
                                            Value('int', 'num_dimensions'), 
                                            Value('int', 'num_clusters'),
                                            Value('int', 'num_events'),
                                            Value('float*', 'd_likelihoods') ]),
                        Block([Statement(s) for s in estep2_statements]) )
        mstep_N_launch_func = FunctionBody(
                        FunctionDeclaration(Value('void', 'mstep_N_launch'),
                                        [   Value('float*', 'd_fcs_data_by_event'), 
                                            Value('clusters_t*', 'd_clusters'), 
                                            Value('int', 'num_dimensions'), 
                                            Value('int', 'num_clusters'),
                                            Value('int', 'num_events') ]),
                        Block([Statement(s) for s in mstep_N_statements]) )
        mstep_means_launch_func = FunctionBody(
                        FunctionDeclaration(Value('void', 'mstep_means_launch'),
                                        [   Value('float*', 'd_fcs_data_by_dimension'), 
                                            Value('clusters_t*', 'd_clusters'), 
                                            Value('int', 'num_dimensions'), 
                                            Value('int', 'num_clusters'),
                                            Value('int', 'num_events') ]),
                        Block([Statement(s) for s in mstep_means_statements]) )
        mstep_covar_launch_funcs = [ FunctionBody(
                        FunctionDeclaration(Value('void', 'mstep_covar_launch_' + v),
                                        [   Value('float*', 'd_fcs_data_by_dimension'), 
                                            Value('float*', 'd_fcs_data_by_event'), 
                                            Value('clusters_t*', 'd_clusters'), 
                                            Value('int', 'num_dimensions'), 
                                            Value('int', 'num_clusters'),
                                            Value('int', 'num_events'),
                                            Value('float*', 'temp_buffer_2b') ]),
                        Block([Statement(s) for s in mstep_covar_launch_versions[v]]) ) for v in version_suffix_list ]

        for x in [  seed_clusters_launch_func, 
                    constants_kernel_launch_func, 
                    estep1_launch_func, 
                    estep2_launch_func, 
                    mstep_N_launch_func, 
                    mstep_means_launch_func ]:
            cuda_mod.add_function(x) 
        for x in mstep_covar_launch_funcs:
            cuda_mod.add_function(x)

        # Setup toolchain and compile

        def pyublas_inc():
            file, pathname, descr = find_module("pyublas")
            return join(pathname, "..", "include")
        def numpy_inc():
            file, pathname, descr = find_module("numpy")
            return join(pathname, "core", "include")
        nvcc_toolchain = aspmod.nvcc_toolchain
        nvcc_toolchain.cflags += ["-arch=sm_20"]
        aspmod.toolchain.add_library("project",['.','./include',pyublas_inc(),numpy_inc()],[],[])
        nvcc_toolchain.add_library("project",['.','./include'],[],[])
        aspmod.toolchain.add_library("cutils",['/home/henry/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/henry/NVIDIA_GPU_Computing_SDK/C/lib','/home/henry/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])
        nvcc_toolchain.add_library("cutils",['/home/henry/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/henry/NVIDIA_GPU_Computing_SDK/C/lib','/home/henry/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])

        aspmod.compile()

        #main( int device, int original_num_clusters, int num_dimensions, int num_events, pyublas::numpy_array<float> input_data )
        aspmod.train( 0, self.M, self.D, self.N, self.X)
        means = aspmod.compiled_module.trained.means.reshape((self.M, self.D))
        covars = aspmod.compiled_module.trained.covars.reshape((self.M, self.D, self.D))

        return means, covars
