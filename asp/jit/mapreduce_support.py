from mrjob.job import MRJob

"""
NOTE: this file cannot import anything that won't be in the remote env.
"""

class AspMRJob(MRJob):
    """
    Encapsulates an Asp-specific MapReduce job.
    """
    def mapper(self, key, value):
        yield 0, [int(value) * 2]

    def reducer(self, key, values):
        v = map(lambda x:x, values)
        yield 0, reduce(lambda x,y: x+y, v)

# this appears to be necessary because this script will be called as __main__ on
# every worker node
if __name__ == '__main__':
    job = AspMRJob()
    job.run()
