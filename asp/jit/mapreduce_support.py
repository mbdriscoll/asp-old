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
    def job_runner_kwargs(self):
        """
        General configuration options.
        """
        config = super(AspMRJob, self).job_runner_kwargs()
        config['hadoop_input_format'] = "org.apache.hadoop.mapred.lib.NLineInputFormat"
        config['jobconf']['mapred.line.input.format.linespermap'] = 28
        config['cmdenv']["LD_LIBRARY_PATH"] = '/global/homes/d/driscoll/carver/opt/local/lib'
        config['python_bin'] = "/global/homes/d/driscoll/carver/opt/local/bin/python"
        return config

    def emr_job_runner_kwargs(self):
        """
        Elastic MapReduce specific configuration options.
        """
        config = super(AspMRJob, self).emr_job_runner_kwargs()
        config['bootstrap_scripts'] += ["s3://speechdiarizer32/deploy.sh"]
        config['setup_cmds'] += ["export PATH=/home/hadoop/opt/local/bin:$PATH"]
        config['setup_cmds'] += ["export LD_LIBRARY_PATH=/home/hadoop/opt/local/lib:$LD_LIBRARY_PATH"]
        config['python_bin'] = '/home/hadoop/opt/local/bin/python'
        return config

    def hadoop_job_runner_kwargs(self):
        """
        Hadoop specific configuration options.
        """
        config = super(AspMRJob, self).hadoop_job_runner_kwargs()
        #config['hadoop_extra_args'] += ["--verbose"]
        return config

# this appears to be necessary because this script will be called as __main__ on
# every worker node.
if __name__ == '__main__':
    AspMRJob().run()
