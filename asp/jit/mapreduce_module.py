from asp.config import MapReduceDetector
from scala_module import PseudoModule # steal this for now


class MapReduceModule:
    """
    A module to support specialization via MapReduce. Does nothing for now,
    but exists to be consistent with other backends.
    """
    pass


class MapReduceToolchain:
    """
    Tools to execute mapreduce jobs.
    """
    def __init__(self, cluster='hadoop'):
        MapReduceDetector.detect_or_exit(cluster)
        self.cluster = cluster # (local|hadoop|emr)


# TODO this should probably subclass ASPBackend but I can't the imports right.
# For now just override the same methods.
class MapReduceBackend(object):
    """
    Class to encapsulate a mapreduce backend for Asp.
    """
    def __init__(self, module=None, toolchain=None):
        self.module = module or MapReduceModule()
        self.toolchain = toolchain or MapReduceToolchain()

    def compile(self):
        """
        Trigger a compile of this backend.
        """
        raise NotImplementedError

    def get_compiled_function(self, name):
        """
        Return a callable for a raw compiled function (that is, this must be a
        variant name rather than a function name). Note that for
        MapReduceBackends functions are not compiled, just stored.
        """
        try:
            func = getattr(self.compiled_module, name)
        except:
            raise AttributeError("Function %s not found in compiled module." %
                                 (name,))
        return func

    def specialize(self, AspMRJobCls):
        """
        Return a callable that runs the given map and reduce functions.
        """
        from asp.jit.mapreduce_support import AspMRJob
        from sys import stderr

        def mr_callable(args):
            mr_args = ['-v', '-r', self.toolchain.cluster]
            job = AspMRJobCls(args=mr_args).sandbox(stdin=args)
            runner = job.make_runner()
            runner.run()
            kv_pairs = map(job.parse_output_line, runner.stream_output())
            return kv_pairs

        return mr_callable
