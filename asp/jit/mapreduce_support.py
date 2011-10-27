from sys import stderr
from mrjob.job import MRJob

"""
NOTE: this file cannot import anything that won't be in the remote env.
"""

"""
logging is enabled by mrjob, and configuring it is required if we're printing
debug info to stderr.
"""
import logging
logging.basicConfig()

class AspMRJob(MRJob):
    """
    Encapsulates an Asp-specific MapReduce job.
    """
    def mapper(self, key, value):
        #print >>stderr, "MAP", key, value
        yield 0, [float(value) * 2]

    def reducer(self, key, values):
        val = reduce(lambda x,y: x+y, values)
        #print >>stderr, "REDUCE", key, val
        yield 0, val

# this appears to be necessary because this script will be called as __main__ on
# every worker node
if __name__ == '__main__':
    AspMRJob().run()
