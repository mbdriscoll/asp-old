from sys import stderr
from mrjob.job import MRJob

"""
NOTE: this file cannot import anything that won't be in the remote env.
"""

# Logging is enabled by mrjob, and configuring it is required if we're printing
# debug info to stderr.
import logging
logging.basicConfig()

class AspMRJob(MRJob):
    """
    Encapsulates an Asp-specific MapReduce job.
    """

    def mapper(self, key, value):
        """
        Doubles value and stores it in a list for easy
        concantenation in the reduce phase.
        """
        yield 1, [float(value) * 2]

    def reducer(self, key, values):
        """
        Concatenates lists in VALUES.
        """
        val = reduce(lambda x,y: x+y, values)
        val.sort() # this is serial => bad
        yield 1, val

    def emr_job_runner_kwargs(self):
        """
        Elastic MapReduce specific configuration options.
        """
        config = super(AspMRJob, self).emr_job_runner_kwargs()
        config['num_ec2_instances'] = 1
        config['ec2_instance_type'] = 'm1.small'
        config['bootstrap_scripts'] += ["s3://speechdiarizer32/deploy.sh"]
        config['setup_cmds'] += ["export PATH=/home/hadoop/opt/local/bin:$PATH"]
        config['setup_cmds'] += ["export LD_LIBRARY_PATH=/home/hadoop/opt/local/lib:$LD_LIBRARY_PATH"]
        config['python_bin'] = '/home/hadoop/opt/local/bin/python'
        return config

# this appears to be necessary because this script will be called as __main__ on
# every worker node.
if __name__ == '__main__':
    AspMRJob().run()
