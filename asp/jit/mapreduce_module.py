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
    def __init__(self, cluster='local'):
        if not MapReduceDetector.detect(cluster):
            raise EnvironmentError("Cannot detect MapReduce platform: %s" %\
                                   cluster)
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
        pass

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

    def specialize(self, fname, mapper, reducer):
        return lambda x: reduce(reducer, map(mapper, x))
