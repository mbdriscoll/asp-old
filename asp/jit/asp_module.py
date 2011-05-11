import codepy, codepy.jit, codepy.toolchain, codepy.bpl
from asp.util import *
import asp.codegen.cpp_ast as cpp_ast
import pickle
from variant_history import *

class HelperMethodInfo(object):
    def __init__(self, func_name, module_name):
        self.func_name = func_name
        self.module_name = module_name

class InternalModule(object):

    def __init__(self, name, cache_dir, codepy_module, codepy_toolchain, nvcc_toolchain=None, compilable=True):
        self.name = name
        self.dirty = False
	self.compilable = compilable # Will be false for the boost module used by cuda
        self.cache_dir = cache_dir
        self.codepy_module = codepy_module
        self.codepy_toolchain = codepy_toolchain
        self.nvcc_toolchain = nvcc_toolchain

    def compile(self):
        if not self.compilable: return None
        if self.nvcc_toolchain:
	    return self.codepy_module.compile(self.codepy_toolchain, self.nvcc_toolchain, cache_dir=self.cache_dir)
        else:
	    return self.codepy_module.compile(self.codepy_toolchain, cache_dir=self.cache_dir)
        self.dirty = False

class ASPModule(object):

    def __init__(self, use_cuda=False, use_cilk=False):
        self.specialized_methods = {}
        self.helper_methods = {}
        self.cache_dir = "cache"
        self.modules = {} 
        self.compiled_modules = {} 
        self.modules['base'] = InternalModule('base', self.cache_dir, codepy.bpl.BoostPythonModule(), codepy.toolchain.guess_toolchain())
        if use_cuda:
            self.modules['cuda_boost'] = InternalModule('cuda_boost', self.cache_dir, codepy.bpl.BoostPythonModule(), codepy.toolchain.guess_toolchain(), compilable=False)
            self.modules['cuda'] = InternalModule('cuda', self.cache_dir, codepy.cuda.CudaModule(self.modules['cuda_boost'].codepy_module), codepy.toolchain.guess_toolchain(), codepy.toolchain.guess_nvcc_toolchain())
            self.modules['cuda'].codepy_module.add_to_preamble([cpp_ast.Include('cuda.h', False)])
        if use_cilk:
            self.modules['cilk'] = InternalModule('cilk', self.cache_dir, codepy.bpl.BoostPythonModule(), codepy.toolchain.guess_toolchain())
            #self.modules[cilk].codepy_module.add_to_preamble([cpp_ast.Include('cilk.h', False)])

    def add_library(self, feature, include_dirs, library_dirs=[], libraries=[], name='base'):
        self.modules[name].codepy_toolchain.add_library(feature, include_dirs, library_dirs, libraries)

    def add_cuda_arch_spec(self, arch):
        archflag = '-arch='
        if 'sm_' not in arch: archflag += 'sm_' 
        archflag += arch
        self.modules['cuda'].codepy_toolchain.cflags += [archflag]

    def add_header(self, include_file, module_name='base'):
        self.modules[module_name].codepy_module.add_to_preamble([cpp_ast.Include(include_file, False)])

    def add_to_preamble(self, pa, module_name='base'):
        if isinstance(pa, str):
            pa = [cpp_ast.Line(pa)]
        self.modules[module_name].codepy_module.add_to_preamble(pa)

    def add_to_init(self, stmt, module_name='base'):
        if isinstance(stmt, str):
            stmt = [cpp_ast.Line(stmt)]
        self.modules[module_name].codepy_module.add_to_init(stmt)

    def add_to_module(self, block, module_name='base'):
        if isinstance(block, str):
            block = [cpp_ast.Line(block)]
        self.modules[module_name].codepy_module.add_to_module(block)

    def get_name_from_func(self, func):
        """
        returns the name of a function from a CodePy FunctionBody object
        """
        return func.fdecl.subdecl.name

    def add_function_helper(self, func, fname=None, module_name='base'):
        if isinstance(func, str):
            if fname == None:
                raise Exception("Cannot add a function as a string without specifying the function's name")
            self.modules[module_name].codepy_module.add_to_module([cpp_ast.Line(func)])
            self.modules[module_name].codepy_module.add_to_init([cpp_ast.Statement(
                        "boost::python::def(\"%s\", &%s)" % (fname, fname))])
        else:
            self.modules[module_name].codepy_module.add_function(func)
        self.modules[module_name].dirty = True
        if module_name == 'cuda_boost': self.modules['cuda'].dirty = True

    def add_function_with_variants(self, variant_funcs, func_name, variant_names, key_maker=lambda name, *args, **kwargs: (name), normalizer=lambda results, time: time, limit_funcs=None, compilable=None, param_names=None, module_name='base'):
        limit_funcs = limit_funcs or [lambda name, *args, **kwargs: True]*len(variant_names) 
        compilable = compilable or [True]*len(variant_names)
        param_names = param_names or ['Unknown']*len(variant_names)
        method_info = self.specialized_methods.get(func_name, None)
        if not method_info:
            method_info = CodeVariants(variant_names, key_maker, normalizer, param_names, [module_name]*len(variant_names))
            method_info.limiter.append(variant_names, limit_funcs, compilable)
        else:
            method_info.append(variant_names, [module_name]*len(variant_names))
            method_info.database.clear_oracle()
            method_info.limiter.append(variant_names, limit_funcs, compilable)
        for x in range(0,len(variant_funcs)):
            self.add_function_helper(variant_funcs[x], fname=variant_names[x], module_name=module_name)
        self.specialized_methods[func_name] = method_info

    def add_function(self, funcs, fname=None, variant_names=None, module_name='base'):
        """
        self.add_function(func) takes func as either a generable AST or a string, or
        list of variants in either format.
        """
        if variant_names:
            self.add_function_with_variants(funcs, fname, variant_names, module_name=module_name)
        else:
            variant_funcs = [funcs]
            if not fname:
                fname = self.get_name_from_func(funcs)
            variant_names = [fname]
            self.add_function_with_variants(variant_funcs, fname, variant_names, module_name=module_name)

    def add_helper_function(self, func_name, module_name='base'):
        method_info = self.helper_methods.get(func_name, None)
        if method_info:
            raise Exception("Overwrote helper function info; duplicate method name!")
        else:
            method_info = HelperMethodInfo(func_name, module_name)
        self.add_function_helper("", fname=func_name, module_name=module_name)
        self.helper_methods[func_name] = method_info
     
    def specialized_func(self, name):
        import time
        def special(*args, **kwargs):
            method_info = self.specialized_methods[name]
            key = method_info.make_key(name,*args,**kwargs)
            v_id = method_info.selector.get_v_id_to_run(method_info.v_id_set, key,*args,**kwargs)
            if not v_id: 
                raise Exception("No variant of method found to run on input size %s on the specified device" % str(args))
           
            module_name = method_info.get_module_for_v_id(v_id)
            module_name = 'cuda' if module_name == 'cuda_boost' else module_name
            module = self.compiled_modules[module_name]
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
            module_name = 'cuda' if method_info.module_name == 'cuda_boost' else imethod_info.module_name
            real_func = self.compiled_modules[module_name].__getattribute__(name)
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

    def compile_module(self, module_name):
        self.compiled_modules[module_name] = self.modules[module_name].compile()

    def compile_all(self):
        for name, module in filter(lambda x: x[1].dirty and x[1].compilable, self.modules.iteritems()):
            self.compiled_modules[name] = module.compile()
        
    def __getattr__(self, name):
        if name in self.specialized_methods:
            self.compile_all()
            return self.specialized_func(name)
        elif name in self.helper_methods:
            self.compile_all()
            return self.helper_func(name)
        else:
            raise AttributeError("No method %s found; did you add it?" % name)

