import codepy, codepy.jit, codepy.toolchain, codepy.bpl
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast
import pickle
from variant_history import *

class HelperMethodInfo(object):
    def __init__(self, func_name, backend_name):
        self.func_name = func_name
        self.backend_name = backend_name

class InternalModule(object):

    def __init__(self, name, cache_dir, boost_module, boost_toolchain, extension_module=None, extension_toolchain=None, primary=True):
        self.name = name
        self.dirty = False
        self.cache_dir = cache_dir
        if primary and extension_module: # 
	    self.compilable = True
            self.boost_module = boost_module
            self.boost_toolchain = boost_toolchain
            self.extension_module = extension_module
            self.extension_toolchain = extension_toolchain
            self.codepy_module = extension_module
            self.codepy_toolchain = extension_toolchain
        elif primary and not extension_module:
	    self.compilable = True 
            self.codepy_module = boost_module
            self.codepy_toolchain = boost_toolchain
        else: #not primary
            self.compilable = False
            self.codepy_module = boost_module
            self.codepy_toolchain = boost_toolchain

    def compile(self):
        if not self.compilable: return None
        if self.name == 'cilk':
            #Deal with linking in functions with boost compiled with gcc
            host_code = str(self.boost_module.generate()) + "\n"
            #device_code = str(self.extension_module.generate()) + "\n"
            from codepy.cgen import Line, Module
            body = self.extension_module.preamble + [Line()] + self.extension_module.mod_body
            device_code = str(Module(body)) + "\n"

            host_toolchain = self.boost_toolchain.copy()
            host_toolchain.add_library('cilk',[],['/opt/intel/composerxe_mic.0.042/compiler/lib/intel64','/opt/intel/composerxe_mic.0.042/mpirt/lib/intel64','/opt/intel/composerxe_mic.0.042/ipp/../compiler/lib/intel64','/opt/intel/composerxe_mic.0.042/ipp/lib/intel64','/opt/intel/composerxe_mic.0.042/compiler/lib/intel64','/opt/intel/composerxe_mic.0.042/mkl/lib/intel64','/opt/intel/composerxe_mic.0.042/tbb/lib/intel64//cc4.1.0_libc2.4_kernel2.6.16.21'],['cilkrts','irc','svml'])

            from codepy.jit import compile_from_string, extension_from_string
            from codepy.jit import link_extension
            host_mod_name, host_object, host_compiled = compile_from_string(
               host_toolchain, self.boost_module.name, host_code,
               object=True)
            device_mod_name, device_object, device_compiled = compile_from_string(
               self.extension_toolchain, 'cilk', device_code, 'cilk.cpp',
               object=True)
            if host_compiled or device_compiled:
               ret = link_extension(host_toolchain,
                                     [host_object, device_object],
                                     host_mod_name)
            else:
              import os.path
              destination_base, first_object = os.path.split(host_object)
              module_path = os.path.join(destination_base, host_mod_name
                                         + host_toolchain.so_ext)
              try:
                  from imp import load_dynamic
                  ret = load_dynamic(host_mod_name, module_path)
              except:
                  ret = link_extension(host_toolchain,
                                        [host_object, device_object],
                                        host_mod_name)
            print repr(ret)
            return ret
        elif self.name =='cuda':
	    return self.extension_module.compile(self.boost_toolchain, self.extension_toolchain, cache_dir=self.cache_dir)
        else:
	    return self.codepy_module.compile(self.codepy_toolchain, cache_dir=self.cache_dir)
        self.dirty = False

class ASPModule(object):

    def __init__(self, use_cuda=False, use_cilk=False):
        self.specialized_methods = {}
        self.helper_methods = {}
        self.cache_dir = "cache"
        self.backends = {} 
        self.compiled_modules = {} 
        self.backends['base'] = InternalModule('base', self.cache_dir, codepy.bpl.BoostPythonModule(), codepy.toolchain.guess_toolchain())
        if use_cuda:
            self.backends['cuda_boost'] = InternalModule('cuda_boost', self.cache_dir, codepy.bpl.BoostPythonModule(), codepy.toolchain.guess_toolchain(), primary=False)
            self.backends['cuda'] = InternalModule('cuda', self.cache_dir, self.backends['cuda_boost'].codepy_module, self.backends['cuda_boost'].codepy_toolchain, codepy.cuda.CudaModule(self.backends['cuda_boost'].codepy_module), codepy.toolchain.guess_nvcc_toolchain())
            self.backends['cuda'].codepy_module.add_to_preamble([cpp_ast.Include('cuda.h', False)])
        if use_cilk:
            self.backends['cilk_boost'] = InternalModule('cilk_boost', self.cache_dir, codepy.bpl.BoostPythonModule(), codepy.toolchain.guess_toolchain(), primary=False)
            self.backends['cilk'] = InternalModule('cilk', self.cache_dir, self.backends['cilk_boost'].codepy_module, self.backends['cilk_boost'].codepy_toolchain, codepy.bpl.BoostPythonModule(), codepy.toolchain.guess_toolchain())
            self.backends['cilk'].codepy_module.add_to_preamble([cpp_ast.Include('cilk/cilk.h', False)])
            self.backends['cilk'].codepy_toolchain.cc = 'icc'
            self.backends['cilk'].codepy_toolchain.cflags = ['-O2','-gcc', '-ip']

    def add_library(self, feature, include_dirs, library_dirs=[], libraries=[], name='base'):
        self.backends[name].codepy_toolchain.add_library(feature, include_dirs, library_dirs, libraries)

    def add_cuda_arch_spec(self, arch):
        archflag = '-arch='
        if 'sm_' not in arch: archflag += 'sm_' 
        archflag += arch
        self.backends['cuda'].codepy_toolchain.cflags += [archflag]

    def add_header(self, include_file, backend_name='base'):
        self.backends[backend_name].codepy_module.add_to_preamble([cpp_ast.Include(include_file, False)])

    def add_to_preamble(self, pa, backend_name='base'):
        if isinstance(pa, str):
            pa = [cpp_ast.Line(pa)]
        self.backends[backend_name].codepy_module.add_to_preamble(pa)

    def add_to_init(self, stmt, backend_name='base'):
        if isinstance(stmt, str):
            stmt = [cpp_ast.Line(stmt)]
        self.backends[backend_name].codepy_module.add_to_init(stmt)

    def add_to_module(self, block, backend_name='base'):
        if isinstance(block, str):
            block = [cpp_ast.Line(block)]
        self.backends[backend_name].codepy_module.add_to_module(block)

    def get_name_from_func(self, func):
        """
        returns the name of a function from a CodePy FunctionBody object
        """
        return func.fdecl.subdecl.name

    def add_function_helper(self, fbody, fname=None, backend_name='base'):
        if isinstance(fbody, str):
            if fname == None:
                raise Exception("Cannot add a function as a string without specifying the function's name")
            self.backends[backend_name].codepy_module.add_to_module([cpp_ast.Line(fbody)])
            self.backends[backend_name].codepy_module.add_to_init([cpp_ast.Statement(
                        "boost::python::def(\"%s\", &%s)" % (fname, fname))])
        else:
            self.backends[backend_name].codepy_module.add_function(fbody)
        self.backends[backend_name].dirty = True
        if backend_name.endswith('_boost'): self.backends[backend_name[:-6]].dirty = True

    def add_function_with_variants(self, variant_bodies, func_name, variant_names, key_maker=lambda name, *args, **kwargs: (name), normalizer=lambda results, time: time, limit_funcs=None, compilable=None, param_names=None, backend_name='base'):
        limit_funcs = limit_funcs or [lambda name, *args, **kwargs: True]*len(variant_names) 
        compilable = compilable or [True]*len(variant_names)
        param_names = param_names or ['Unknown']*len(variant_names)
        method_info = self.specialized_methods.get(func_name, None)
        if not method_info:
            method_info = CodeVariants(variant_names, key_maker, normalizer, param_names, [backend_name]*len(variant_names))
            method_info.limiter.append(variant_names, limit_funcs, compilable)
        else:
            method_info.append(variant_names, [backend_name]*len(variant_names))
            method_info.database.clear_oracle()
            method_info.limiter.append(variant_names, limit_funcs, compilable)
        for x in range(0,len(variant_bodies)):
            self.add_function_helper(variant_bodies[x], fname=variant_names[x], backend_name=backend_name)
        self.specialized_methods[func_name] = method_info

    def add_function(self, funcs, fname=None, variant_names=None, backend_name='base'):
        """
        self.add_function(func) takes func as either a generable AST or a string, or
        list of variants in either format.
        """
        if variant_names:
            self.add_function_with_variants(funcs, fname, variant_names, backend_name=backend_name)
        else:
            variant_funcs = [funcs]
            if not fname:
                fname = self.get_name_from_func(funcs)
            variant_names = [fname]
            self.add_function_with_variants(variant_funcs, fname, variant_names, backend_name=backend_name)

    def add_helper_function(self, func_name, backend_name='base'):
        method_info = self.helper_methods.get(func_name, None)
        if method_info:
            raise Exception("Overwrote helper function info; duplicate method name!")
        else:
            method_info = HelperMethodInfo(func_name, backend_name)
        self.add_function_helper("", fname=func_name, backend_name=backend_name)
        self.helper_methods[func_name] = method_info
     
    def specialized_func(self, name):
        import time
        def special(*args, **kwargs):
            method_info = self.specialized_methods[name]
            key = method_info.make_key(name,*args,**kwargs)
            v_id = method_info.selector.get_v_id_to_run(method_info.v_id_set, key,*args,**kwargs)
            if not v_id: 
                raise Exception("No variant of method found to run on input size %s on the specified device" % str(args))
           
            backend_name = method_info.get_module_for_v_id(v_id)
            backend_name = backend_name[:-6] if backend_name.endswith('_boost') else backend_name
            module = self.compiled_modules[backend_name]
            real_func = module.__getattribute__(v_id)
            start_time = time.time() 
            results = real_func(*args, **kwargs)
            elapsed = time.time() - start_time
            value_to_put_in_database = method_info.normalize_performance(results, elapsed)
            method_info.database.add_time( key, value_to_put_in_database, v_id, method_info.v_id_set)
            return results
        return special

    def helper_func(self, name):
        def helper(*args, **kwargs):
            method_info = self.helper_methods[name]
            backend_name = method_info.backend_name[:-6] if method_info.backend_name.endswith('_boost') else method_info.backend_name
            real_func = self.compiled_modules[backend_name].__getattribute__(name)
            return real_func(*args, **kwargs)
        return helper

    def save_method_timings(self, name, file_name=None):
        method_info = self.specialized_methods[name]
        f = open(file_name or self.cache_dir+'/'+name+'.vardump', 'w')
        d = method_info.get_picklable_obj()
        d.update(method_info.database.get_picklable_obj())
        pickle.dump( d, f)
        f.close()

    def restore_method_timings(self, name, file_name=None):
        method_info = self.specialized_methods[name]
        try: 
	    f = open(file_name or self.cache_dir+'/'+name+'.vardump', 'r')
            obj = pickle.load(f)
            if obj: method_info.set_from_pickled_obj(obj)
            if obj: method_info.database.set_from_pickled_obj(obj, method_info.v_id_set)
            f.close()
        except IOError: pass

    def clear_method_timings(self, name):
        method_info = self.specialized_methods[name]
        method_info.database.clear()

    def compile_module(self, backend_name):
        self.compiled_modules[backend_name] = self.backends[backend_name].compile()

    def compile_all(self):
        for name, backend in filter(lambda x: x[1].dirty and x[1].compilable, self.backends.iteritems()):
            self.compiled_modules[name] = backend.compile()
        
    def __getattr__(self, name):
        if name in self.specialized_methods:
            self.compile_all()
            return self.specialized_func(name)
        elif name in self.helper_methods:
            self.compile_all()
            return self.helper_func(name)
        else:
            raise AttributeError("No method %s found; did you add it?" % name)

