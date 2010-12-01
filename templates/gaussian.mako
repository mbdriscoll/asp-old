/*
 * Gaussian Mixture Model Clustering with CUDA
 *
 * Orginal Author: Andrew Pangborn
 * Department of Computer Engineering
 * Rochester Institute of Technology
 * 
 */

#define PI  3.1415926535897931
#define COVARIANCE_DYNAMIC_RANGE 1E6
#define NUM_BLOCKS_ESTEP   ${num_blocks_estep} // Num of blocks per cluster for the E-step
#define NUM_THREADS_ESTEP  ${num_threads_estep} // should be a power of 2 
#define NUM_THREADS_MSTEP  ${num_threads_mstep} // should be a power of 2
#define NUM_EVENT_BLOCKS   ${num_event_blocks}
#define MAX_NUM_DIMENSIONS ${max_num_dimensions}
#define MAX_NUM_CLUSTERS   ${max_num_clusters}
#define DEVICE             ${device_id} // Which GPU to use, if more than 1
#define DIAG_ONLY          ${diag_only} // Using only diagonal covariance matrix, thus all dimensions are considered independent
#define MAX_ITERS          ${max_iters} // Maximum number of iterations for the EM convergence loop
#define MIN_ITERS          ${min_iters}// Minimum number of iterations (normally 0 unless doing performance testing)
#define ENABLE_CODEVAR_2B_BUFFER_ALLOC ${enable_2b_buffer}
#define VERSION_SUFFIX     ${version_suffix}

typedef struct return_array_container
{
  pyublas::numpy_array<float> means;
  pyublas::numpy_array<float> covars;
} ret_arr_con_t;

ret_arr_con_t ret;

// Function prototypes
void writeCluster(FILE* f, clusters_t clusters, int c,  int num_dimensions);
void printCluster(clusters_t clusters, int c, int num_dimensions);
float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);
void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions);
void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions);
void invert_cpu(float* data, int actualsize, float* log_determinant);
int invert_matrix(float* a, int n, float* determinant);

int train (
        int device,
        int num_clusters, 
        int num_dimensions, 
        int num_events, 
        pyublas::numpy_array<float> input_data ) 
{

  float* fcs_data_by_event = input_data.data();
  int original_num_clusters = num_clusters; //should just %s/original_num/num/g
    
  // Set the device to run on... 0 for GTX 480, 1 for GTX 285 on oak
  int GPUCount;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&GPUCount));
  if(GPUCount == 0) {
    //printf("Only 1 CUDA device found, defaulting to it.\n");
    device = 0;
  } else if (GPUCount >= 1 && device >= 0) {
    //printf("Multiple CUDA devices found, selecting device based on user input: %d\n",device);
  } else if(GPUCount >= 1 && DEVICE < GPUCount) {
    //printf("Multiple CUDA devices found, selecting based on compiled default: %d\n",DEVICE);
    device = DEVICE;
  } else {
    //printf("Fatal Error: Unable to set device to %d, not enough GPUs.\n",DEVICE);
    exit(2);
  }
  CUDA_SAFE_CALL(cudaSetDevice(device));
    
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  //printf("\nUsing device - %s\n\n", prop.name);
    
  // Transpose the event data (allows coalesced access pattern in E-step kernel)
  // This has consecutive values being from the same dimension of the data 
  // (num_dimensions by num_events matrix)
  float* fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);
  if(!fcs_data_by_dimension) {
    printf("ERROR, not enough memory for both formats\n");
    return 1;
  }
    
  //printf("Number of events: %d\n",num_events);
  //printf("Number of dimensions: %d\n",num_dimensions);

  for(int e=0; e<num_events; e++) {
    for(int d=0; d<num_dimensions; d++) {
      fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
    }
  }    

  //printf("Number of clusters: %d\n\n",num_clusters);
    
   
  // Setup the cluster data structures on host
  clusters_t clusters;
  clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
  clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
  if(!clusters.means || !clusters.R || !clusters.Rinv || !clusters.memberships) { 
    printf("ERROR: Could not allocate memory for clusters.\n"); 
    return 1; 
  }

  float *temp_buffer_2b = NULL;
#if ENABLE_CODEVAR_2B_BUFFER_ALLOC
  //scratch space to clear out clusters->R
  float *zeroR_2b = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  for(int i = 0; i<num_dimensions*num_dimensions*original_num_clusters; i++) {
    zeroR_2b[i] = 0.0f;
  }
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_buffer_2b),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters, cudaMemcpyHostToDevice) );
#endif
    
  // Declare another set of clusters for saving the results of the best configuration
  clusters_t saved_clusters;
  saved_clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
  saved_clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
  saved_clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
  saved_clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
  saved_clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
  saved_clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  saved_clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  saved_clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
  if(!saved_clusters.means || !saved_clusters.R || !saved_clusters.Rinv || !saved_clusters.memberships) { 
    printf("ERROR: Could not allocate memory for clusters.\n"); 
    return 1; 
  }

  // Setup the cluster data structures on host
  clusters_t scratch_clusters;
  scratch_clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);
  scratch_clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
  scratch_clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
  scratch_clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
  scratch_clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);
  scratch_clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  scratch_clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  scratch_clusters.memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
  if(!scratch_clusters.means || !scratch_clusters.R || !scratch_clusters.Rinv || !scratch_clusters.memberships) { 
    printf("ERROR: Could not allocate memory for scratch_clusters.\n"); 
    return 1; 
  }
  
  // Setup the cluster data structures on device
  // First allocate structures on the host, CUDA malloc the arrays
  // Then CUDA malloc structures on the device and copy them over
  clusters_t temp_clusters;
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.N),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.pi),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.constant),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.avgvar),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.means),sizeof(float)*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.R),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.Rinv),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.memberships),sizeof(float)*num_events*original_num_clusters));
   
  // Allocate a struct on the device 
  clusters_t* d_clusters;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_clusters, sizeof(clusters_t)));
    
  // Copy Cluster data to device
  CUDA_SAFE_CALL(cudaMemcpy(d_clusters,&temp_clusters,sizeof(clusters_t),cudaMemcpyHostToDevice));

  int mem_size = num_dimensions*num_events*sizeof(float);
    
  float min_rissanen, rissanen;
    
  // allocate device memory for FCS data
  float* d_fcs_data_by_event;
  float* d_fcs_data_by_dimension;
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));
  // copy FCS to device
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size,cudaMemcpyHostToDevice) );
    
   
  //////////////// Initialization done, starting kernels //////////////// 
  fflush(stdout);

  // seed_clusters sets initial pi values, 
  // finds the means / covariances and copies it to all the clusters
  seed_clusters_launch( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, num_events);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Seed Kernel execution failed: ");
   
  // Computes the R matrix inverses, and the gaussian constant
  constants_kernel_launch(d_clusters,original_num_clusters,num_dimensions);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Constants Kernel execution failed: ");
    
  // Calculate an epsilon value
  float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.0001;
  int iters;
  float likelihood, old_likelihood;
  // Used to hold the result from regroup kernel
  float* likelihoods = (float*) malloc(sizeof(float)*NUM_BLOCKS_ESTEP);
  float* d_likelihoods;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*NUM_BLOCKS_ESTEP));
    
  // Variables for GMM reduce order
  float distance, min_distance = 0.0;
  int min_c1, min_c2;
  float* d_c;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_c, sizeof(float)));

  //int num_clusters = original_num_clusters;
  //for(int num_clusters=original_num_clusters; num_clusters >= stop_number; num_clusters--) {
    /*************** EM ALGORITHM *****************************/
        
    // do initial regrouping
    // Regrouping means calculate a cluster membership probability
    // for each event and each cluster. Each event is independent,
    // so the events are distributed to different blocks 
    // (and hence different multiprocessors)

  //================================== EM INITIALIZE =======================

    estep1_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods,num_clusters);
    //cudaThreadSynchronize();

    estep2_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
    cudaThreadSynchronize();

    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    // Copy the likelihood totals from each block, sum them up to get a total
    CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS_ESTEP,cudaMemcpyDeviceToHost));
    likelihood = 0.0;
    for(int i=0;i<NUM_BLOCKS_ESTEP;i++) {
     likelihood += likelihoods[i]; 
    }
    //printf("Starter Likelihood: %e\n",likelihood);

    float change = epsilon*2;

    //================================= EM BEGIN ==================================
    //printf("Performing EM algorithm on %d clusters.\n",num_clusters);
    iters = 0;

    // This is the iterative loop for the EM algorithm.
    // It re-estimates parameters, re-computes constants, and then regroups the events
    // These steps keep repeating until the change in likelihood is less than some epsilon        
    while(iters < MIN_ITERS || (iters < MAX_ITERS && fabs(change) > epsilon)) {
      old_likelihood = likelihood;
            
      //params = M step
      // This kernel computes a new N, pi isn't updated until compute_constants though
      mstep_N_launch(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);
      cudaThreadSynchronize();

      // This kernel computes new means
      mstep_means_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);
      cudaThreadSynchronize();
            
#if ENABLE_CODEVAR_2B_BUFFER_ALLOC
      CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(float)*num_dimensions*num_dimensions*num_clusters, cudaMemcpyHostToDevice) );
#endif
      // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
      mstep_covar_launch_${version_suffix}(d_fcs_data_by_dimension,d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events,temp_buffer_2b);
      cudaThreadSynchronize();
                 
      CUT_CHECK_ERROR("M-step Kernel execution failed: ");


      // Inverts the R matrices, computes the constant, normalizes cluster probabilities
      constants_kernel_launch(d_clusters,num_clusters,num_dimensions);
      cudaThreadSynchronize();
      CUT_CHECK_ERROR("Constants Kernel execution failed: ");

      //regroup = E step
      // Compute new cluster membership probabilities for all the events
      estep1_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods,num_clusters);

      estep2_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
      cudaThreadSynchronize();

      CUT_CHECK_ERROR("E-step Kernel execution failed: ");
        
      // check if kernel execution generated an error
      CUT_CHECK_ERROR("Kernel execution failed");
        
      CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS_ESTEP,cudaMemcpyDeviceToHost));
      likelihood = 0.0;
      for(int i=0;i<NUM_BLOCKS_ESTEP;i++) {
        likelihood += likelihoods[i]; 
      }
            
      change = likelihood - old_likelihood;
      //printf("Iter %d likelihood = %f\n", iters, likelihood);
      //printf("Change in likelihood: %f (vs. %f)\n",change, epsilon);

      iters++;

    }//EM Loop
        
    // copy clusters from the device
    CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters, d_clusters, sizeof(clusters_t),cudaMemcpyDeviceToHost));
    // copy all of the arrays from the structs
    CUDA_SAFE_CALL(cudaMemcpy(clusters.N, temp_clusters.N, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.pi, temp_clusters.pi, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.constant, temp_clusters.constant, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.avgvar, temp_clusters.avgvar, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.means, temp_clusters.means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.R, temp_clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.Rinv, temp_clusters.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(clusters.memberships, temp_clusters.memberships, sizeof(float)*num_events*num_clusters,cudaMemcpyDeviceToHost));
        
    //} // outer loop from M to 1 clusters

  ret.means = pyublas::numpy_array<float>(num_dimensions*num_clusters);
  std::copy( clusters.means, clusters.means+num_dimensions*num_clusters, ret.means.begin());
  ret.covars = pyublas::numpy_array<float>(num_dimensions*num_dimensions*num_clusters);
  std::copy( clusters.R, clusters.R+num_dimensions*num_dimensions*num_clusters, ret.covars.begin());

  //================================ EM DONE ==============================
  //printf("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,num_clusters);
  //printf("DONE COMPUTING\n");
 
  // cleanup host memory
  //free(fcs_data_by_event); NOW OWNED BY PYTHON!
  free(fcs_data_by_dimension);
  free(clusters.N);
  free(clusters.pi);
  free(clusters.constant);
  free(clusters.avgvar);
  free(clusters.means);
  free(clusters.R);
  free(clusters.Rinv);
  free(clusters.memberships);

  free(saved_clusters.N);
  free(saved_clusters.pi);
  free(saved_clusters.constant);
  free(saved_clusters.avgvar);
  free(saved_clusters.means);
  free(saved_clusters.R);
  free(saved_clusters.Rinv);
  free(saved_clusters.memberships);
    
  free(scratch_clusters.N);
  free(scratch_clusters.pi);
  free(scratch_clusters.constant);
  free(scratch_clusters.avgvar);
  free(scratch_clusters.means);
  free(scratch_clusters.R);
  free(scratch_clusters.Rinv);
  free(scratch_clusters.memberships);
   
  free(likelihoods);

  // cleanup GPU memory
  CUDA_SAFE_CALL(cudaFree(d_likelihoods));
 
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_event));
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_dimension));

  CUDA_SAFE_CALL(cudaFree(temp_clusters.N));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.pi));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.constant));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.avgvar));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.means));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.R));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.Rinv));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.memberships));
  CUDA_SAFE_CALL(cudaFree(d_clusters));

  return 0;
}

void writeCluster(FILE* f, clusters_t clusters, int c, int num_dimensions) {
  fprintf(f,"Probability: %f\n", clusters.pi[c]);
  fprintf(f,"N: %f\n",clusters.N[c]);
  fprintf(f,"Means: ");
  for(int i=0; i<num_dimensions; i++){
    fprintf(f,"%.3f ",clusters.means[c*num_dimensions+i]);
  }
  fprintf(f,"\n");

  fprintf(f,"\nR Matrix:\n");
  for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
      fprintf(f,"%.3f ", clusters.R[c*num_dimensions*num_dimensions+i*num_dimensions+j]);
    }
    fprintf(f,"\n");
  }
  fflush(f);   
  /*
    fprintf(f,"\nR-inverse Matrix:\n");
    for(int i=0; i<num_dimensions; i++) {
    for(int j=0; j<num_dimensions; j++) {
    fprintf(f,"%.3f ", c->Rinv[i*num_dimensions+j]);
    }
    fprintf(f,"\n");
    } 
  */
}

void printCluster(clusters_t clusters, int c, int num_dimensions) {
  writeCluster(stdout,clusters,c,num_dimensions);
}

float cluster_distance(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
  // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster
  add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);
    
  return clusters.N[c1]*clusters.constant[c1] + clusters.N[c2]*clusters.constant[c2] - temp_cluster.N[0]*temp_cluster.constant[0];
}

void add_clusters(clusters_t clusters, int c1, int c2, clusters_t temp_cluster, int num_dimensions) {
  float wt1,wt2;
 
  wt1 = (clusters.N[c1]) / (clusters.N[c1] + clusters.N[c2]);
  wt2 = 1.0f - wt1;
    
  // Compute new weighted means
  for(int i=0; i<num_dimensions;i++) {
    temp_cluster.means[i] = wt1*clusters.means[c1*num_dimensions+i] + wt2*clusters.means[c2*num_dimensions+i];
  }
    
  // Compute new weighted covariance
  for(int i=0; i<num_dimensions; i++) {
    for(int j=i; j<num_dimensions; j++) {
      // Compute R contribution from cluster1
      temp_cluster.R[i*num_dimensions+j] = ((temp_cluster.means[i]-clusters.means[c1*num_dimensions+i])
                                            *(temp_cluster.means[j]-clusters.means[c1*num_dimensions+j])
                                            +clusters.R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
      // Add R contribution from cluster2
      temp_cluster.R[i*num_dimensions+j] += ((temp_cluster.means[i]-clusters.means[c2*num_dimensions+i])
                                             *(temp_cluster.means[j]-clusters.means[c2*num_dimensions+j])
                                             +clusters.R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
      // Because its symmetric...
      temp_cluster.R[j*num_dimensions+i] = temp_cluster.R[i*num_dimensions+j];
    }
  }
    
  // Compute pi
  temp_cluster.pi[0] = clusters.pi[c1] + clusters.pi[c2];
    
  // compute N
  temp_cluster.N[0] = clusters.N[c1] + clusters.N[c2];

  float log_determinant;
  // Copy R to Rinv matrix
  memcpy(temp_cluster.Rinv,temp_cluster.R,sizeof(float)*num_dimensions*num_dimensions);
  // Invert the matrix
  invert_cpu(temp_cluster.Rinv,num_dimensions,&log_determinant);
  // Compute the constant
  temp_cluster.constant[0] = (-num_dimensions)*0.5*logf(2*PI)-0.5*log_determinant;
    
  // avgvar same for all clusters
  temp_cluster.avgvar[0] = clusters.avgvar[0];
}

void copy_cluster(clusters_t dest, int c_dest, clusters_t src, int c_src, int num_dimensions) {
  dest.N[c_dest] = src.N[c_src];
  dest.pi[c_dest] = src.pi[c_src];
  dest.constant[c_dest] = src.constant[c_src];
  dest.avgvar[c_dest] = src.avgvar[c_src];
  memcpy(&(dest.means[c_dest*num_dimensions]),&(src.means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
  memcpy(&(dest.R[c_dest*num_dimensions*num_dimensions]),&(src.R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  memcpy(&(dest.Rinv[c_dest*num_dimensions*num_dimensions]),&(src.Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  // do we need to copy memberships?
}

static float double_abs(float x);

static int 
ludcmp(float *a,int n,int *indx,float *d);

static void 
lubksb(float *a,int n,int *indx,float *b);

/*
 * Inverts a square matrix (stored as a 1D float array)
 * 
 * actualsize - the dimension of the matrix
 *
 * written by Mike Dinolfo 12/98
 * version 1.0
 */
void invert_cpu(float* data, int actualsize, float* log_determinant)  {
    int maxsize = actualsize;
    int n = actualsize;
    *log_determinant = 0.0;

    if (actualsize == 1) { // special case, dimensionality == 1
        *log_determinant = logf(data[0]);
        data[0] = 1.0 / data[0];
    } else if(actualsize >= 2) { // dimensionality >= 2
        for (int i=1; i < actualsize; i++) data[i] /= data[0]; // normalize row 0
        for (int i=1; i < actualsize; i++)  { 
            for (int j=i; j < actualsize; j++)  { // do a column of L
                float sum = 0.0;
                for (int k = 0; k < i; k++)  
                    sum += data[j*maxsize+k] * data[k*maxsize+i];
                data[j*maxsize+i] -= sum;
            }
            if (i == actualsize-1) continue;
            for (int j=i+1; j < actualsize; j++)  {  // do a row of U
                float sum = 0.0;
                for (int k = 0; k < i; k++)
                    sum += data[i*maxsize+k]*data[k*maxsize+j];
                data[i*maxsize+j] = 
                    (data[i*maxsize+j]-sum) / data[i*maxsize+i];
            }
        }

        for(int i=0; i<actualsize; i++) {
            *log_determinant += log10(fabs(data[i*n+i]));
            //printf("log_determinant: %e\n",*log_determinant); 
        }
        //printf("\n\n");
        for ( int i = 0; i < actualsize; i++ )  // invert L
            for ( int j = i; j < actualsize; j++ )  {
                float x = 1.0;
                if ( i != j ) {
                    x = 0.0;
                    for ( int k = i; k < j; k++ ) 
                        x -= data[j*maxsize+k]*data[k*maxsize+i];
                }
                data[j*maxsize+i] = x / data[j*maxsize+j];
            }
        for ( int i = 0; i < actualsize; i++ )   // invert U
            for ( int j = i; j < actualsize; j++ )  {
                if ( i == j ) continue;
                float sum = 0.0;
                for ( int k = i; k < j; k++ )
                    sum += data[k*maxsize+j]*( (i==k) ? 1.0 : data[i*maxsize+k] );
                data[i*maxsize+j] = -sum;
            }
        for ( int i = 0; i < actualsize; i++ )   // final inversion
            for ( int j = 0; j < actualsize; j++ )  {
                float sum = 0.0;
                for ( int k = ((i>j)?i:j); k < actualsize; k++ )  
                    sum += ((j==k)?1.0:data[j*maxsize+k])*data[k*maxsize+i];
                data[j*maxsize+i] = sum;
            }
    } else {
        printf("Error: Invalid dimensionality for invert(...)\n");
    }
 }


/*
 * Another matrix inversion function
 * This was modified from the 'cluster' application by Charles A. Bouman
 */
int invert_matrix(float* a, int n, float* determinant) {
    int  i,j,f,g;
   
    float* y = (float*) malloc(sizeof(float)*n*n);
    float* col = (float*) malloc(sizeof(float)*n);
    int* indx = (int*) malloc(sizeof(int)*n);
    /*
    printf("\n\nR matrix before LU decomposition:\n");
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            printf("%.2f ",a[i*n+j]);
        }
        printf("\n");
    }*/

    *determinant = 0.0;
    if(ludcmp(a,n,indx,determinant)) {
        printf("Determinant mantissa after LU decomposition: %f\n",*determinant);
        printf("\n\nR matrix after LU decomposition:\n");
        for(i=0; i<n; i++) {
            for(j=0; j<n; j++) {
                printf("%.2f ",a[i*n+j]);
            }
            printf("\n");
        }
       
      for(j=0; j<n; j++) {
        *determinant *= a[j*n+j];
      }
     
      printf("determinant: %E\n",*determinant);
     
      for(j=0; j<n; j++) {
        for(i=0; i<n; i++) col[i]=0.0;
        col[j]=1.0;
        lubksb(a,n,indx,col);
        for(i=0; i<n; i++) y[i*n+j]=col[i];
      }

      for(i=0; i<n; i++)
      for(j=0; j<n; j++) a[i*n+j]=y[i*n+j];
     
      printf("\n\nMatrix at end of clust_invert function:\n");
      for(f=0; f<n; f++) {
          for(g=0; g<n; g++) {
              printf("%.2f ",a[f*n+g]);
          }
          printf("\n");
      }
      free(y);
      free(col);
      free(indx);
      return(1);
    }
    else {
        *determinant = 0.0;
        free(y);
        free(col);
        free(indx);
        return(0);
    }
}

static float double_abs(float x)
{
       if(x<0) x = -x;
       return(x);
}

#define TINY 1.0e-20

static int
ludcmp(float *a,int n,int *indx,float *d)
{
    int i,imax,j,k;
    float big,dum,sum,temp;
    float *vv;

    vv= (float*) malloc(sizeof(float)*n);
   
    *d=1.0;
   
    for (i=0;i<n;i++)
    {
        big=0.0;
        for (j=0;j<n;j++)
            if ((temp=fabsf(a[i*n+j])) > big)
                big=temp;
        if (big == 0.0)
            return 0; /* Singular matrix  */
        vv[i]=1.0/big;
    }
       
   
    for (j=0;j<n;j++)
    {  
        for (i=0;i<j;i++)
        {
            sum=a[i*n+j];
            for (k=0;k<i;k++)
                sum -= a[i*n+k]*a[k*n+j];
            a[i*n+j]=sum;
        }
       
        /*
        int f,g;
        printf("\n\nMatrix After Step 1:\n");
        for(f=0; f<n; f++) {
            for(g=0; g<n; g++) {
                printf("%.2f ",a[f*n+g]);
            }
            printf("\n");
        }*/
       
        big=0.0;
        dum=0.0;
        for (i=j;i<n;i++)
        {
            sum=a[i*n+j];
            for (k=0;k<j;k++)
                sum -= a[i*n+k]*a[k*n+j];
            a[i*n+j]=sum;
            dum=vv[i]*fabsf(sum);
            //printf("sum: %f, dum: %f, big: %f\n",sum,dum,big);
            //printf("dum-big: %E\n",fabs(dum-big));
            if ( (dum-big) >= 0.0 || fabs(dum-big) < 1e-3)
            {
                big=dum;
                imax=i;
                //printf("imax: %d\n",imax);
            }
        }
       
        if (j != imax)
        {
            for (k=0;k<n;k++)
            {
                dum=a[imax*n+k];
                a[imax*n+k]=a[j*n+k];
                a[j*n+k]=dum;
            }
            *d = -(*d);
            vv[imax]=vv[j];
        }
        indx[j]=imax;
       
        /*
        printf("\n\nMatrix after %dth iteration of LU decomposition:\n",j);
        for(f=0; f<n; f++) {
            for(g=0; g<n; g++) {
                printf("%.2f ",a[f][g]);
            }
            printf("\n");
        }
        printf("imax: %d\n",imax);
        */


        /* Change made 3/27/98 for robustness */
        if ( (a[j*n+j]>=0)&&(a[j*n+j]<TINY) ) a[j*n+j]= TINY;
        if ( (a[j*n+j]<0)&&(a[j*n+j]>-TINY) ) a[j*n+j]= -TINY;

        if (j != n-1)
        {
            dum=1.0/(a[j*n+j]);
            for (i=j+1;i<n;i++)
                a[i*n+j] *= dum;
        }
    }
    free(vv);
    return(1);
}

#undef TINY

static void
lubksb(float *a,int n,int *indx,float *b)
{
    int i,ii,ip,j;
    float sum;

    ii = -1;
    for (i=0;i<n;i++)
    {
        ip=indx[i];
        sum=b[ip];
        b[ip]=b[i];
        if (ii >= 0)
            for (j=ii;j<i;j++)
                sum -= a[i*n+j]*b[j];
        else if (sum)
            ii=i;
        b[i]=sum;
    }
    for (i=n-1;i>=0;i--)
    {
        sum=b[i];
        for (j=i+1;j<n;j++)
            sum -= a[i*n+j]*b[j];
        b[i]=sum/a[i*n+i];
    }
}

