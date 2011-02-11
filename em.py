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

class Components(object):
    
    def __init__(self, M, D, weights = None, means = None, covars = None):

        self.M = M
        self.D = D
        self.weights = weights or np.empty(M, dtype=np.float32)
        self.means = means or np.empty(M*D, dtype=np.float32)
        self.covars = covars or np.empty(M*D*D, dtype=np.float32)

    def init_random_weights(self):
        return numpy.random.random((self.M))

    def init_random_means(self):
        return numpy.random.random((self.M,self.D))

    def init_random_covars(self):
        return numpy.random.random((self.M, self.D, self.D))

    def shrink_components(self, new_M):
        self.weights = np.resize(self.weights, new_M) #= np.delete(self.weights, np.s_[new_M:])
        self.means = np.resize(self.means, new_M*self.D) #= np.delete(self.means, np.s_[new_M*self.D:])
        self.covars = np.resize(self.covars, new_M*self.D*self.D) #= np.delete(self.covars, np.s_[new_M*self.D*self.D:])

class EvalData(object):

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.memberships = np.zeros((M,N), dtype=np.float32)
        self.loglikelihoods = np.zeros(N, dtype=np.float32)
        self.likelihood = 0.0

    def resize(self, N, M):
        self.memberships.resize((M,N))
        self.memberships = np.ascontiguousarray(self.memberships)
        self.loglikelihoods.resize(N)
        self.loglikelihoods = np.ascontiguousarray(self.loglikelihoods)
        self.M = M
        self.N = N

class GMM(object):

    #Singleton ASP mode shared by all instances of GMM
    asp_mod = None    
    def get_asp_mod(self): return GMM.asp_mod or self.initialize_asp_mod()

    #Default parameter space for code variants
    variant_param_default = {
            'num_blocks_estep': ['16'],
            'num_threads_estep': ['512'],
            'num_threads_mstep': ['256'],
            'num_event_blocks': ['128'],
            'max_num_dimensions': ['50'],
            'max_num_components': ['128'],
            'diag_only': ['0'],
            'max_iters': ['10'],
            'min_iters': ['1'],
            'covar_version_name': ['V1', 'V2A', 'V2B', 'V3']
    }

    #Flags to keep track of memory allocations, singletons
    event_data_gpu_copy = None
    event_data_cpu_copy = None
    component_data_gpu_copy = None
    component_data_cpu_copy = None
    eval_data_gpu_copy = None
    eval_data_cpu_copy = None

    # nternal functions to allocate and deallocate component and event data on the CPU and GPU
    def internal_alloc_event_data(self, X):
        #TODO: test for not null
        #if not X.any(): return
        if not np.array_equal(GMM.event_data_gpu_copy, X):
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

    def internal_alloc_component_data(self):
        #TODO: test for not null
        #if not self.components.weights.size: return
        if GMM.component_data_gpu_copy != self.components:
            if GMM.component_data_gpu_copy:
                self.internal_free_component_data()
            self.get_asp_mod().alloc_components_on_GPU(self.M, self.D)
            self.get_asp_mod().alloc_components_on_CPU(self.M, self.D, self.components.weights, self.components.means, self.components.covars)
            self.get_asp_mod().copy_component_data_CPU_to_GPU(self.M, self.D)
            GMM.component_data_gpu_copy = self.components
            GMM.component_data_cpu_copy = self.components
            
    def internal_free_component_data(self):
        if GMM.component_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_components_on_GPU()
            GMM.component_data_gpu_copy = None
        if GMM.component_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_components_on_CPU()
            GMM.component_data_cpu_copy = None

    def internal_alloc_eval_data(self, X):
        if self.eval_data.M != self.M or self.eval_data.N != X.shape[0] or GMM.eval_data_gpu_copy != self.eval_data:
            if GMM.eval_data_gpu_copy is not None:
                self.internal_free_eval_data()
            self.eval_data.resize(X.shape[0], self.M)
            self.get_asp_mod().alloc_evals_on_GPU(X.shape[0], self.M)
            self.get_asp_mod().alloc_evals_on_CPU(self.eval_data.memberships, self.eval_data.loglikelihoods)
            GMM.eval_data_gpu_copy = self.eval_data
            GMM.eval_data_cpu_copy = self.eval_data
            
    def internal_free_eval_data(self):
        if GMM.eval_data_gpu_copy is not None:
            self.get_asp_mod().dealloc_evals_on_GPU()
            GMM.eval_data_gpu_copy = None
        if GMM.eval_data_cpu_copy is not None:
            self.get_asp_mod().dealloc_evals_on_CPU()
            GMM.eval_data_cpu_copy = None

    def __init__(self, M, D, variant_param_space=None, device=0, means=None, covars=None, weights=None):
        self.device = device
        self.M = M
        self.D = D
        self.variant_param_space = variant_param_space or GMM.variant_param_default
        self.components = Components(M, D, weights, means, covars)
        self.eval_data = EvalData(1, M)
        self.clf = None # pure python mirror module

    #Called the first time a GMM instance tries to use a specialized function
    def initialize_asp_mod(self):

        # Create ASP module
        GMM.asp_mod = asp_module.ASPModule(use_cuda=True)
        cuda_mod = GMM.asp_mod.cuda_module

        #Add decls to preamble necessary for linking to compiled CUDA sources
        component_t_decl =""" 
            typedef struct components_struct {
                float* N;        // expected # of pixels in component: [M]
                float* pi;       // probability of component in GMM: [M]
                float* constant; // Normalizing constant [M]
                float* avgvar;    // average variance [M]
                float* means;   // Spectral mean for the component: [M*D]
                float* R;      // Covariance matrix: [M*D*D]
                float* Rinv;   // Inverse of covariance matrix: [M*D*D]
            } components_t;"""
        GMM.asp_mod.add_to_preamble(component_t_decl)
        cuda_mod.add_to_preamble([Line(component_t_decl)])

        #Add necessary headers
        host_system_header_names = [ 'stdlib.h', 'stdio.h', 'string.h', 'math.h', 'time.h','pyublas/numpy.hpp' ]
        for x in host_system_header_names: GMM.asp_mod.add_to_preamble([Include(x, True)])
        cuda_mod.add_to_preamble([Include('stdio.h',True)])
        #TODO: Figure out whether we can free ourselves from cutils
        host_project_header_names = [ 'cutil_inline.h'] 
        for x in host_project_header_names: GMM.asp_mod.add_to_preamble([Include(x, False)])

        #Add C/CUDA source code that is not based on code variant parameters
        #TODO: stop using templates and just read from file?
        #TODO: also, rename files and make them .c and .cu instead of .mako?
        c_base_tpl = AspTemplate.Template(filename="templates/gaussian.mako")
        c_base_rend  = c_base_tpl.render()
        GMM.asp_mod.module.add_to_module([Line(c_base_rend)])
        cu_base_tpl = AspTemplate.Template(filename="templates/theta_kernel_base.mako")
        cu_base_rend = cu_base_tpl.render()
        cuda_mod.add_to_module([Line(cu_base_rend)])

        #Add C/CUDA source code that is based on code variant parameters
        c_train_tpl = AspTemplate.Template(filename="templates/train.mako")
        c_eval_tpl = AspTemplate.Template(filename="templates/eval.mako")
        cu_kern_tpl = AspTemplate.Template(filename="templates/theta_kernel.mako")
        c_decl_tpl = AspTemplate.Template(filename="templates/gaussian_decl.mako") 

        def render_and_add_to_module( param_dict ):
            keys = param_dict.keys()
            keys.sort()
            vals = map(param_dict.get, keys) #gets vals based on alpha order of keys
            c_train_rend  = c_train_tpl.render( param_val_list = vals, **param_dict)
            c_eval_rend  = c_eval_tpl.render( param_val_list = vals, **param_dict)
            cu_kern_rend = cu_kern_tpl.render( param_val_list = vals, **param_dict)
            c_decl_rend  = c_decl_tpl.render( param_val_list = vals, **param_dict)

            cuda_mod.add_to_module([Line(cu_kern_rend)])
            GMM.asp_mod.add_to_preamble(c_decl_rend)
            GMM.asp_mod.add_function_with_variants( [c_train_rend], 
                                                    "train", 
                                                    [ 'train_'+'_'.join(vals) ],
                                                    lambda name, *args, **kwargs: (name, args[0], args[1], args[2]),
                                                    keys
                                                  )
            GMM.asp_mod.add_function_with_variants( [c_eval_rend], 
                                                    "eval", 
                                                    [ 'eval_'+'_'.join(vals) ],
                                                    lambda name, *args, **kwargs: (name, args[0], args[1], args[2]),
                                                    keys
                                                  )

        def generate_permutations ( key_arr, val_arr_arr, current, add_func):
            idx = len(current)
            name = key_arr[idx]
            for v in val_arr_arr[idx]:
                current[name]  = v
                if idx == len(key_arr)-1:
                    add_func(current)
                else:
                    generate_permutations(key_arr, val_arr_arr, current, add_func)
            del current[name]

        generate_permutations( self.variant_param_space.keys(), self.variant_param_space.values(), {}, render_and_add_to_module)

        # Add set GPU device function
        GMM.asp_mod.add_function("", fname="set_GPU_device")
        
        # Add malloc, copy and free functions
        GMM.asp_mod.add_function("", fname="alloc_events_on_CPU")
        GMM.asp_mod.add_function("", fname="alloc_events_on_GPU")
        GMM.asp_mod.add_function("", fname="alloc_components_on_CPU")
        GMM.asp_mod.add_function("", fname="alloc_components_on_GPU")
        GMM.asp_mod.add_function("", fname="alloc_evals_on_CPU")
        GMM.asp_mod.add_function("", fname="alloc_evals_on_GPU")
        GMM.asp_mod.add_function("", fname="copy_event_data_CPU_to_GPU")
        GMM.asp_mod.add_function("", fname="copy_component_data_CPU_to_GPU")
        GMM.asp_mod.add_function("", fname="copy_component_data_GPU_to_CPU")
        GMM.asp_mod.add_function("", fname="copy_evals_data_GPU_to_CPU")
        GMM.asp_mod.add_function("", fname="dealloc_events_on_CPU")
        GMM.asp_mod.add_function("", fname="dealloc_events_on_GPU")
        GMM.asp_mod.add_function("", fname="dealloc_components_on_CPU")
        GMM.asp_mod.add_function("", fname="dealloc_components_on_GPU")
        GMM.asp_mod.add_function("", fname="dealloc_temp_components_on_CPU")
        GMM.asp_mod.add_function("", fname="dealloc_evals_on_CPU")
        GMM.asp_mod.add_function("", fname="dealloc_evals_on_GPU")

        # Add getter functions
        GMM.asp_mod.add_function("", fname="get_temp_component_pi")
        GMM.asp_mod.add_function("", fname="get_temp_component_means")
        GMM.asp_mod.add_function("", fname="get_temp_component_covars")
        
        # Add merge components function
        GMM.asp_mod.add_function("", fname="relink_components_on_CPU")
        GMM.asp_mod.add_function("", fname="compute_distance_rissanen")
        GMM.asp_mod.add_function("", fname="merge_components")

        #Add Boost interface links for components and distance objects
        GMM.asp_mod.add_to_init("""boost::python::class_<components_struct>("Components");
            boost::python::scope().attr("components") = boost::python::object(boost::python::ptr(&components));""")
        GMM.asp_mod.add_to_init("""boost::python::class_<return_component_container>("ReturnClusterContainer")
            .def(pyublas::by_value_rw_member( "new_component", &return_component_container::component))
            .def(pyublas::by_value_rw_member( "distance", &return_component_container::distance)) ;
            boost::python::scope().attr("component_distance") = boost::python::object(boost::python::ptr(&ret));""")
        
        # Setup toolchain and compile
        def pyublas_inc():
            file, pathname, descr = find_module("pyublas")
            return join(pathname, "..", "include")
        def numpy_inc():
            file, pathname, descr = find_module("numpy")
            return join(pathname, "core", "include")

        nvcc_toolchain = GMM.asp_mod.nvcc_toolchain
        nvcc_toolchain.cflags += ["-arch=sm_13" if self.device else "-arch=sm_20"]
        GMM.asp_mod.toolchain.add_library("project",['.','./include',pyublas_inc(),numpy_inc()],[],[])
        nvcc_toolchain.add_library("project",['.','./include'],[],[])

        #TODO: Get rid of awful hardcoded paths necessitaty by cutils
        GMM.asp_mod.toolchain.add_library("cutils",['/home/egonina/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/henry/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/egonina/NVIDIA_GPU_Computing_SDK/C/lib','/home/egonina/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])
        nvcc_toolchain.add_library("cutils",['/home/egonina/NVIDIA_GPU_Computing_SDK/C/common/inc','/home/egonina/NVIDIA_GPU_Computing_SDK/C/shared/inc'],['/home/egonina/NVIDIA_GPU_Computing_SDK/C/lib','/home/egonina/NVIDIA_GPU_Computing_SDK/shared/lib'],['cutil_x86_64', 'shrutil_x86_64'])

        GMM.asp_mod.set_GPU_device(self.device)
        
        #print GMM.asp_mod.module.generate()
        GMM.asp_mod.compile()
        return GMM.asp_mod

    def __del__(self):
        self.internal_free_event_data()
        self.internal_free_component_data()
        self.internal_free_eval_data()
    
    def train_using_python(self, input_data):
        from scikits.learn import mixture
        self.clf = mixture.GMM(n_states=self.M, cvtype='full')
        self.clf.fit(input_data)
        return self.clf.means, self.clf.covars
    
    def eval_using_python(self, obs_data):
        from scikits.learn import mixture
        if self.clf is not None:
            return self.clf.eval(obs_data)
        else: return []

    def predict_using_python(self, obs_data):
        from scikits.learn import mixture
        if self.clf is not None:
            return self.clf.predict(obs_data)
        else: return []

    def train(self, input_data):
        N = input_data.shape[0] #TODO: handle types other than np.array?
        if input_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (input_data.shape[1], self.D)
        self.internal_alloc_event_data(input_data)
        self.internal_alloc_eval_data(input_data)
        self.internal_alloc_component_data()
        self.eval_data.likelihood = self.get_asp_mod().train(self.M, self.D, N, input_data)
        return self

    def eval(self, obs_data):
        N = obs_data.shape[0]
        if obs_data.shape[1] != self.D:
            print "Error: Data has %d features, model expects %d features." % (obs_data.shape[1], self.D)
        self.internal_alloc_event_data(obs_data)
        self.internal_alloc_eval_data(obs_data)
        self.eval_data.likelihood = self.get_asp_mod().eval(self.M, self.D, N, obs_data)
        logprob = self.eval_data.loglikelihoods
        posteriors = self.eval_data.memberships
        return logprob, posteriors # N log probabilities, NxM posterior probabilities for each component

    def score(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return logprob # N log probabilities

    def decode(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return logprob, posteriors.argmax(axis=0) # N log probabilities, N indexes of most likely components 

    def predict(self, obs_data):
        logprob, posteriors = self.eval(obs_data)
        return posteriors.argmax(axis=0) # N indexes of most likely components

    def merge_components(self, min_c1, min_c2, min_component):
        self.get_asp_mod().dealloc_temp_components_on_CPU()
        self.get_asp_mod().merge_components(min_c1, min_c2, min_component, self.M, self.D)
        self.M -= 1
        self.components.shrink_components(self.M)
        self.get_asp_mod().relink_components_on_CPU(self.components.weights, self.components.means, self.components.covars)
        return 

    def compute_distance_rissanen(self, c1, c2):
        self.get_asp_mod().compute_distance_rissanen(c1, c2, self.D)
        new_component = self.get_asp_mod().compiled_module.component_distance.new_component
        dist = self.get_asp_mod().compiled_module.component_distance.distance
        return new_component, dist

    def get_new_component_means(self, new_component):
        return self.get_asp_mod().get_temp_component_means(new_component, self.D).reshape((1, self.D))

    def get_new_component_covars(self, new_component):
        return self.get_asp_mod().get_temp_component_covars(new_component, self.D).reshape((1, self.D, self.D))
    
def compute_distance_BIC(gmm1, gmm2, data):
    cd1_M = gmm1.M
    cd2_M = gmm2.M
    nComps = cd1_M + cd2_M

    ratio1 = float(cd1_M)/float(nComps)
    ratio2 = float(cd2_M)/float(nComps)

    w = np.append(ratio1*gmm1.components.weights, ratio2*gmm2.components.weights)
    m = np.append(gmm1.components.means, gmm2.components.means)
    c = np.append(gmm1.components.covarss, gmm2.components.covars)
    temp_GMM = GMM(nComps, gmm1.D, w, m, c)

    temp_GMM.train(data)

    return temp_GMM.eval_data.likelihood - (gmm1.eval_data.likelihood + gmm2.eval_data.likelihood)

