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
import pylab as pl
import matplotlib as mpl
import itertools

# Create data

#input_data = np.recfromcsv('test.csv', names=None, dtype=np.float32)

# Handle Inputs

#TODO
#printf("Usage: %s num_clusters infile outfile [target_num_clusters] [device]\n",argv[0]);
#  if(argc <= 6 && argc >= 4) {
#    if(*num_clusters < 1) {
#    FILE* infile = fopen(argv[2],"r");
#    if(!infile) {
#      if(*target_num_clusters > *num_clusters) {

#TODO: More params current in gaussian.h? Such as MIN_ITERS and MAX_ITERS

n = 300
N = 2*n
D = 2
M = 2

np.random.seed(0)
C = np.array([[0., -0.7], [3.5, .7]])
Y = np.r_[
    np.dot(np.random.randn(n, 2), C),
    np.random.randn(n, 2) + np.array([5, 5]),
    ]
X = np.array(Y, dtype=np.float32)

version = sys.argv[1]
version_define_mapping = {'1' : 'CODEVAR_1', 
                          '2' : 'CODEVAR_2A',
                          '2A': 'CODEVAR_2A',
                          '2B': 'CODEVAR_2B',
                          '3' : 'CODEVAR_3A',
                          '3A': 'CODEVAR_3A' }
num_event_blocks = 128

# Render C/CUDA source templates based on inputs

c_main_tpl = AspTemplate.Template(filename="templates/gaussian.mako")
cu_kern_tpl = AspTemplate.Template(filename="templates/theta_kernel.mako")
c_main_rend = c_main_tpl.render()
cu_kern_rend = cu_kern_tpl.render()

# Create host-side module

aspmod = asp_module.ASPModule(use_cuda=True)

host_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h','pyublas/numpy.hpp' ]
host_project_header_names = [ 'cutil_inline.h', 'gaussian.h'] 

aspmod.add_to_init("""boost::python::class_<return_array_container>("ReturnArrayContainer")
    .def(pyublas::by_value_rw_member(
         "means",
         &return_array_container::means))
    .def(pyublas::by_value_rw_member(
         "covars",
         &return_array_container::covars))
    ;
    boost::python::scope().attr("trained") = boost::python::object(boost::python::ptr(&ret));""")
for x in host_system_header_names: aspmod.add_to_preamble([Include(x, True)])
for x in host_project_header_names: aspmod.add_to_preamble([Include(x, False)])
aspmod.add_to_preamble([Define( version_define_mapping[version], 1)])
aspmod.add_function(c_main_rend, fname="main")
aspmod.module.mod_body.append(Line(open('invert_matrix.cpp', 'r').read()))

# Create cuda-device module

cuda_mod = aspmod.cuda_module
cuda_mod.add_to_preamble([Include('stdio.h')])
cuda_mod.add_to_preamble([Include('gaussian.h')])
cuda_mod.add_to_preamble([Define( version_define_mapping[version], 1)])
cuda_mod.add_to_preamble([Define( "NUM_EVENT_BLOCKS", int(num_event_blocks))])
cuda_mod.add_to_module([Line(cu_kern_rend)])

seed_clusters_statements = [ 'seed_clusters<<< 1, NUM_THREADS_MSTEP >>>( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, num_events);' ]
constants_kernel_statements = [ 'constants_kernel<<<original_num_clusters, 64>>>(d_clusters,original_num_clusters,num_dimensions);' ]
estep1_statements = [ 'estep1<<<dim3(NUM_BLOCKS,num_clusters), NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods);' ]
estep2_statements = [ 'estep2<<<NUM_BLOCKS, NUM_THREADS_ESTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);' ]
mstep_N_statements = [ 'mstep_N<<<num_clusters, NUM_THREADS_MSTEP>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);' ]
mstep_means_statements = ['dim3 gridDim1(num_clusters,num_dimensions);',
      'mstep_means<<<gridDim1, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);' ]
mstep_covar_statements_v1 = [ 'dim3 gridDim2(num_clusters,num_dimensions*(num_dimensions+1)/2);',
      'mstep_covariance<<<gridDim2, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);']
mstep_covar_statements_v2a = [ 'int num_blocks = num_clusters;',
      'int num_threads = num_dimensions*(num_dimensions+1)/2;',
      'mstep_covariance_2a<<<num_clusters, num_threads>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);' ]
mstep_covar_statements_v2b = [ 
      'int num_event_blocks = NUM_EVENT_BLOCKS;',
      'int event_block_size = num_events%NUM_EVENT_BLOCKS == 0 ? num_events/NUM_EVENT_BLOCKS:num_events/(NUM_EVENT_BLOCKS-1);',
      'dim3 gridDim2(num_clusters,num_event_blocks);',
      'int num_threads = num_dimensions*(num_dimensions+1)/2;',
      'mstep_covariance_2b<<<gridDim2, num_threads>>>(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events, event_block_size, num_event_blocks, temp_buffer_2b);' ]
mstep_covar_statements_v3a = ['int num_blocks = num_dimensions*(num_dimensions+1)/2;',
      'mstep_covariance_3a<<<num_blocks, NUM_THREADS_MSTEP>>>(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);' ]

version_launch_mapping = {'1' : {'covar': mstep_covar_statements_v1},
                          '2' : {'covar': mstep_covar_statements_v2a},
                          '2A': {'covar': mstep_covar_statements_v2a},
                          '2B': {'covar': mstep_covar_statements_v2b},
                          '3' : {'covar': mstep_covar_statements_v3a},
                          '3A': {'covar': mstep_covar_statements_v3a} }

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
mstep_covar_launch_func = FunctionBody(
                FunctionDeclaration(Value('void', 'mstep_covar_launch'),
                                [   Value('float*', 'd_fcs_data_by_dimension'), 
                                    Value('float*', 'd_fcs_data_by_event'), 
                                    Value('clusters_t*', 'd_clusters'), 
                                    Value('int', 'num_dimensions'), 
                                    Value('int', 'num_clusters'),
                                    Value('int', 'num_events'),
                                    Value('float*', 'temp_buffer_2b') ]),
                Block([Statement(s) for s in version_launch_mapping[version]['covar']]) )

for x in [  seed_clusters_launch_func, 
            constants_kernel_launch_func, 
            estep1_launch_func, 
            estep2_launch_func, 
            mstep_N_launch_func, 
            mstep_means_launch_func,
            mstep_covar_launch_func ]:
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

#compiled_module = cuda_mod.compile(aspmod.toolchain, nvcc_toolchain, debug=True)
aspmod.compile()

# Run E-M

splot = pl.subplot(111, aspect='equal')
color_iter = itertools.cycle (['r', 'g', 'b', 'c'])

#main( int device, int original_num_clusters, int num_dimensions, int num_events, pyublas::numpy_array<float> input_data )
aspmod.main( 0, M, D, N, X)
means = aspmod.compiled_module.trained.means.reshape((M, D))
print means
covars = aspmod.compiled_module.trained.covars.reshape((M, D, D))
print covars
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


#TODO: Get back cluster descriptions
