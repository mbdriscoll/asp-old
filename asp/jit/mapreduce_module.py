#from asp.config import MapReduceDetector
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
#        if not MapReduceDetector.detect(cluster):
#            raise EnvironmentError("Cannot detect MapReduce platform: %s" %\
#                                   cluster)
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
        """
        Return a callable that runs the given map and reduce functions.
        """

        def callable(*args, **kwargs):
            mr_job = MRWordCounter()
            mr_job.sandbox()
            mr_job.run()

        return callable

# Abstract this away eventually
from mrjob.job import MRJob
class MRWordCounter(MRJob):
    DEFAULT_OUTPUT_PROTOCOL = 'raw_value'
    def mapper(self, key, value):
        yield 0, [int(value) * 2]
    def reducer(self, key, values):
        v = map(lambda x:x, values)
        yield (0, reduce(lambda x,y: x+y, v))

# this appears to be necessary because this script will be called as __main__ on
# every worker node
if __name__ == '__main__':
    job = MRWordCounter()
    job.run()
