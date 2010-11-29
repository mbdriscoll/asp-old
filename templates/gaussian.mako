/*
 * Gaussian Mixture Model Clustering with CUDA
 *
 * Author: Andrew Pangborn
 *
 * Department of Computer Engineering
 * Rochester Institute of Technology
 * 
 */

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

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main (
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
    PRINT("Only 1 CUDA device found, defaulting to it.\n");
    device = 0;
  } else if (GPUCount >= 1 && device >= 0) {
    PRINT("Multiple CUDA devices found, selecting device based on user input: %d\n",device);
  } else if(GPUCount >= 1 && DEVICE < GPUCount) {
    PRINT("Multiple CUDA devices found, selecting based on compiled default: %d\n",DEVICE);
    device = DEVICE;
  } else {
    printf("Fatal Error: Unable to set device to %d, not enough GPUs.\n",DEVICE);
    exit(2);
  }
  CUDA_SAFE_CALL(cudaSetDevice(device));
    
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);
  printf("\nUsing device - %s\n\n", prop.name);
    
  // Transpose the event data (allows coalesced access pattern in E-step kernel)
  // This has consecutive values being from the same dimension of the data 
  // (num_dimensions by num_events matrix)
  float* fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);
  if(!fcs_data_by_dimension) {
    printf("ERROR, not enough memory for both formats\n");
    return 1;
  }
    
  printf("Number of events: %d\n",num_events);
  printf("Number of dimensions: %d\n",num_dimensions);

  for(int e=0; e<num_events; e++) {
    for(int d=0; d<num_dimensions; d++) {
      fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
    }
  }    

  printf("Number of clusters: %d\n\n",num_clusters);
    
   
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
#ifdef CODEVAR_2B
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
  DEBUG("Finished allocating memory on device for clusters.\n");
    
  // Copy Cluster data to device
  CUDA_SAFE_CALL(cudaMemcpy(d_clusters,&temp_clusters,sizeof(clusters_t),cudaMemcpyHostToDevice));
  DEBUG("Finished copying cluster data to device.\n");

  int mem_size = num_dimensions*num_events*sizeof(float);
    
  float min_rissanen, rissanen;
    
  // allocate device memory for FCS data
  float* d_fcs_data_by_event;
  float* d_fcs_data_by_dimension;
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));
  DEBUG("Finished allocating memory on device for clusters.\n");
  // copy FCS to device
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size,cudaMemcpyHostToDevice) );
  DEBUG("Finished copying FCS data to device.\n");
    
   
  //////////////// Initialization done, starting kernels //////////////// 
  DEBUG("Invoking seed_clusters kernel...");
  fflush(stdout);

  // seed_clusters sets initial pi values, 
  // finds the means / covariances and copies it to all the clusters
  seed_clusters_launch( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, num_events);
  cudaThreadSynchronize();
  DEBUG("done.\n"); 
  CUT_CHECK_ERROR("Seed Kernel execution failed: ");
   
  DEBUG("Invoking constants kernel...",num_threads);
  // Computes the R matrix inverses, and the gaussian constant
  constants_kernel_launch(d_clusters,original_num_clusters,num_dimensions);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Constants Kernel execution failed: ");
  DEBUG("done.\n");
    
  // Calculate an epsilon value
  //int ndata_points = num_events*num_dimensions;
  float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.0001;
  float likelihood, old_likelihood;
  int iters;
    
  //epsilon = 1e-6;
  PRINT("Gaussian.cu: epsilon = %f\n",epsilon);

  // Used to hold the result from regroup kernel
  float* likelihoods = (float*) malloc(sizeof(float)*NUM_BLOCKS);
  float* d_likelihoods;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*NUM_BLOCKS));
    
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
    DEBUG("Invoking regroup (E-step) kernel with %d blocks...",NUM_BLOCKS);

    estep1_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods,num_clusters);
    //cudaThreadSynchronize();

    estep2_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
    cudaThreadSynchronize();

    DEBUG("done.\n");
    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Kernel execution failed");

    // Copy the likelihood totals from each block, sum them up to get a total
    CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
    likelihood = 0.0;
    for(int i=0;i<NUM_BLOCKS;i++) {
     likelihood += likelihoods[i]; 
    }
    printf("Starter Likelihood: %e\n",likelihood);

    float change = epsilon*2;

    //================================= EM BEGIN ==================================
    printf("Performing EM algorithm on %d clusters.\n",num_clusters);
    iters = 0;

    // This is the iterative loop for the EM algorithm.
    // It re-estimates parameters, re-computes constants, and then regroups the events
    // These steps keep repeating until the change in likelihood is less than some epsilon        
    while(iters < MIN_ITERS || (iters < MAX_ITERS && fabs(change) > epsilon)) {
      old_likelihood = likelihood;
            
      DEBUG("Invoking reestimate_parameters (M-step) kernel...",num_threads);
      //params = M step
      // This kernel computes a new N, pi isn't updated until compute_constants though
      mstep_N_launch(d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events);
      cudaThreadSynchronize();

      // This kernel computes new means
      mstep_means_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events);
      cudaThreadSynchronize();
            
#ifdef CODEVAR_2B
      CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(float)*num_dimensions*num_dimensions*num_clusters, cudaMemcpyHostToDevice) );
#endif
      // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
      mstep_covar_launch(d_fcs_data_by_dimension,d_fcs_data_by_event,d_clusters,num_dimensions,num_clusters,num_events,temp_buffer_2b);
      cudaThreadSynchronize();
                 
      CUT_CHECK_ERROR("M-step Kernel execution failed: ");
      DEBUG("done.\n");

      DEBUG("Invoking constants kernel...",num_threads);

      // Inverts the R matrices, computes the constant, normalizes cluster probabilities
      constants_kernel_launch(d_clusters,num_clusters,num_dimensions);
      cudaThreadSynchronize();
      CUT_CHECK_ERROR("Constants Kernel execution failed: ");
      DEBUG("done.\n");

      //regroup = E step
      // Compute new cluster membership probabilities for all the events
      estep1_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_events,d_likelihoods,num_clusters);

      estep2_launch(d_fcs_data_by_dimension,d_clusters,num_dimensions,num_clusters,num_events,d_likelihoods);
      cudaThreadSynchronize();

      CUT_CHECK_ERROR("E-step Kernel execution failed: ");
      DEBUG("done.\n");
        
      // check if kernel execution generated an error
      CUT_CHECK_ERROR("Kernel execution failed");
        
      CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*NUM_BLOCKS,cudaMemcpyDeviceToHost));
      likelihood = 0.0;
      for(int i=0;i<NUM_BLOCKS;i++) {
        likelihood += likelihoods[i]; 
      }
            
      change = likelihood - old_likelihood;
      printf("Iter %d likelihood = %f\n", iters, likelihood);
      printf("Change in likelihood: %f (vs. %f)\n",change, epsilon);

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
  printf("DONE COMPUTING\n");
 
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
