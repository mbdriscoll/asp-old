#include <stdio.h>
#include <cuda.h>
#define	CUDASAFECALL(cmd, desc)	{								\
			if((res = (cmd)) != CUDA_SUCCESS){					\
				printf("ERROR: %s [errno=%d]\n", desc, res);	\
				return 1;										\
			}													\
		}

int main(){
	int i, res, count;
	int major, minor;
	char name[100];
	CUdevice dev;

	CUDASAFECALL(cuInit(0), "Init CUDA Driver API");
	CUDASAFECALL(cuDeviceGetCount(&count), "Get number of GPUs w/ Compute Capability >= 1.0");
	for(i=0;i<count;i++){
		CUDASAFECALL(cuDeviceGet(&dev, 0), "Get device handle");
		CUDASAFECALL(cuDeviceGetName(name, 100, dev), "Get device name");
		CUDASAFECALL(cuDeviceComputeCapability(&major, &minor, dev), "Get device compute capability");
		printf("GPU#%d: %s\n", i, name);
		printf("Device Compute Cabability: %d.%d\n", major, minor, dev);
	}

	return 0;
}
