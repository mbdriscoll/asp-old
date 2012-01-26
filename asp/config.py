import re

class CompilerDetector(object):
    """
    Detect if a particular compiler is available by trying to run it.
    """
    def detect(self, compiler):
        import subprocess
        try:
            retcode = subprocess.call([compiler, "--version"])
        except:
            return False

        return (retcode == 0)
        

class PlatformDetector(object):
    def __init__(self):
        self.rawinfo = []

    def get_gpu_nfo(self):
        raise NotImplementedError

    def get_cpu_info(self):
        self.rawinfo = self.read_cpu_info()
        info = {}
        info['numCores'] = self.parse_num_cores()
        info['vendorID'] = self.parse_cpu_info('vendor_id')
        info['model'] = int(self.parse_cpu_info('model'))
        info['cpuFamily'] = int(self.parse_cpu_info('cpu family'))
        info['cacheSize'] = int(self.parse_cpu_info('cache size'))
        info['capabilities'] = self.parse_capabilities()
        return info

    def get_compilers(self):
        return filter(CompilerDetector().detect, ["gcc", "icc", "nvcc"])

    def parse_capabilities(self):
        matcher = re.compile("flags\s+:")
        for line in self.rawinfo:
            if re.match(matcher, line):
                return line.split(":")[1].split(" ")
    
        
    def parse_num_cores(self):
        matcher = re.compile("processor\s+:")
        count = 0
        for line in self.rawinfo:
            if re.match(matcher, line):
                count +=1
        return count
        
    def parse_cpu_info(self, item):
        matcher = re.compile(item +"\s+:\s*(\w+)")
        for line in self.rawinfo:
            if re.match(matcher, line):
                return re.match(matcher, line).group(1)
        
    def read_cpu_info(self):
        return open("/proc/cpuinfo", "r").readlines()


class MapReduceDetector(object):
    """
    Detect if a MapReduce platform is available.
    """
    def __init__(self):
        """ Fail on instantiation. """
        raise RuntimeError("MapReduceDetector should not be instantiated.")

    @classmethod
    def detect(cls, platform):
        """ Detect if a particular platform is available. """
        import os
        try:
            import mrjob
            if (platform == 'emr'):
                aws_access_key_id = os.environ["AWS_ACCESS_KEY_ID"]
                aws_secret_acces_key = os.environ["AWS_SECRET_ACCESS_KEY"]
            elif (platform == 'hadoop'):
                hadoop_home = os.environ["HADOOP_HOME"]
            elif (platform == 'local'):
                pass
            else:
                return False
        except ImportError:
            return False
        except KeyError:
            return False
        return platform

    @classmethod
    def get_platforms(cls):
        """ Returns a list of available MapReduce Platforms. """
        return filter(cls.detect, ['local', 'hadoop', 'emr'])

    @classmethod
    def detect_or_exit(cls, platform):
        if not cls.detect(platform):
            raise EnvironmentError("Unable to detect '%s' MapReduce platform. See configuration instructions for all platforms at http://packages.python.org/mrjob/writing-and-running.html#running-on-emr" % platform)
