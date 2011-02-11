import asp.jit.asp_module as asp_module
import numpy as np
from em import *
import pickle
import sys

param_type_map = {
        'num_blocks_estep': 'cardinal',
        'num_threads_estep': 'cardinal',
        'num_threads_mstep': 'cardinal',
        'num_event_blocks': 'cardinal',
        'max_num_dimensions': 'cardinal',
        'max_num_components': 'cardinal',
        'diag_only': 'binary',
        'max_iters': 'cardinal',
        'min_iters': 'cardinal',
        'covar_version_name': 'nominal'
}

if __name__ == '__main__':  
        ifile_name = sys.argv[1]
        ofile_name = sys.argv[2]
        func_name = sys.argv[3]
        device_id = sys.argv[4]
        gmm = GMM(1,1)
        mod = gmm.get_asp_mod()
        mod.restore_func_variant_timings(func_name,ifile_name)
        var_names = mod.compiled_methods_with_variants[func_name].variant_names
        param_names = mod.compiled_methods_with_variants[func_name].param_names
        var_times = mod.compiled_methods_with_variants[func_name].variant_times
        f = file(ofile_name, 'a')
        f.write("Heading, Function Name, Device Name, Input Params,,,Variant Params,,,,,,,,,,Time\n")
        f.write("Name,function,device,M,D,N,%s,Time\n" % ','.join(param_names))
        f.write("Type,nominal,nominal,cardinal,cardinal,cardinal,%s,real\n" % 
                ','.join([param_type_map[n] for n in param_names]))
        for size, times in var_times.items():
            for name, time in zip(var_names, times):
                f.write(",%s,%s,%s,%s,%s\n" % ( func_name, 
                                                device_id,  
                                                ','.join([str(p) for p in size[1:]]),
                                                ','.join(name.split('_')[1:]),
                                                time ) )
        f.close()

# function,, device,, inputs,, [names from param_names],, time
#  name,, device,, D, M, N,, [params from var name], time
