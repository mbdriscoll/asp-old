from mrjob.protocol import PickleProtocol as protocol
from asp.jit import mapreduce_support as mr
import cPickle as pickle
from stat import *

class FtdockMRJob(mr.AspMRJob):

    DEFAULT_INPUT_PROTOCOL = 'pickle'
    DEFAULT_PROTOCOL = 'pickle'
    
    def job_runner_kwargs(self):
        config = super(FtdockMRJob, self).job_runner_kwargs()
        config['cmdenv']['PYTHONPATH'] = ":".join([
            "/Users/driscoll/sejits/asp",
            "/Users/driscoll/sejits/ftdock_v2.0",
            "/global/homes/d/driscoll/carver/asp",
            "/global/homes/d/driscoll/carver/ftdock_v2.0"
        ])
        config['bootstrap_mrjob'] = False
        config['hadoop_extra_args'] += [
            "-verbose",
            "-mapdebug", "/global/homes/d/driscoll/carver/debug/debugger.sh"
        ]
        return config
    
    def mapper(self, key, value):
        """
        Each mapper executes ftdock for a combination (qi, qj, qk)
        """
        from ftdock_main import ftdock
        dim, arguments = key
        geometry_res = ftdock(dim[0], dim[1], dim[2], *arguments)
        yield 1, geometry_res


class AllCombMap(object):

    def __init__(self, lists_to_combine, ftdock_fxn, *ftdock_args):
        self._lists_to_combine = lists_to_combine
        self._ftdock_fnx = ftdock_fxn
        self._ftdock_args = ftdock_args

    def execute(self, nproc=1):
        cloud_flag = True
        mapfxn = self.ftdock_using_mapreduce if cloud_flag else self.ftdock_classic
        return mapfxn(self._lists_to_combine, self._ftdock_args)

    def ftdock_using_mapreduce(self, lists_to_combine, ftdock_args):
        """
        Perform docking experiment using MapReduce
        """
        print "Map-Reduce execution"
        
        # Add a map task for each point in the search space
        import itertools
        task_args = [protocol.write((dim, ftdock_args), None) for dim in itertools.product(*lists_to_combine)]
    
        import asp.jit.asp_module as asp_module
        mod = asp_module.ASPModule(use_mapreduce=True)
        mod.add_mr_function("ftdock_mr", FtdockMRJob)
        kv_pairs = mod.ftdock_mr(task_args)
        return map(lambda (k,v): v, kv_pairs)
    
    def ftdock_classic(self, lists_to_combine, ftdock_args):
        """
        Perform docking experiment using AllCombMap
        """
        raise NotImplementedError
        """
        print "Classic execution"
        geometry_list = AllCombMap(lists_to_combine, ftdock, *ftdock_args).execute(nproc=2)
        return geometry_list
        """

# this appears to be necessary because this script will be called as __main__ on
# every worker node.
if __name__ == '__main__':
    FtdockMRJob().run()
