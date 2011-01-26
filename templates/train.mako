
float train${'_'+'_'.join(param_val_list)} (
//float train (
                             int num_clusters, 
                             int num_dimensions, 
                             int num_events, 
                             pyublas::numpy_array<float> input_data ) 
{

  //printf("Number of clusters: %d\n\n",num_clusters);
  
  float min_rissanen, rissanen;
  num_scratch_clusters = 0;
  
  //allocate MxM pointers for scratch clusters used during merging
  //TODO: take this out as a separate callable function?
 
  scratch_cluster_arr = (clusters_t**)malloc(sizeof(clusters_t*)*num_clusters*num_clusters);
  
  int original_num_clusters = num_clusters; //should just %s/original_num/num/g

  // ================= Cluster membership alloc on CPU and GPU =============== 
  cluster_memberships = (float*) malloc(sizeof(float)*num_events*original_num_clusters);
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_cluster_memberships),sizeof(float)*num_events*original_num_clusters));
  // ========================================================================= 
  
   
  // ================= Temp buffer for codevar 2b ================ 
  float *temp_buffer_2b = NULL;
%if covar_version_name.upper() in ['2B','V2B','_V2B']:
    //scratch space to clear out clusters->R
    float *zeroR_2b = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
    for(int i = 0; i<num_dimensions*num_dimensions*original_num_clusters; i++) {
        zeroR_2b[i] = 0.0f;
    }
    CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_buffer_2b),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
    CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(float)*num_dimensions*num_dimensions*original_num_clusters, cudaMemcpyHostToDevice) );
%endif
  //=============================================================== 
    
  //////////////// Initialization done, starting kernels //////////////// 
  fflush(stdout);

  // seed_clusters sets initial pi values, 
  // finds the means / covariances and copies it to all the clusters
  seed_clusters_launch${'_'+'_'.join(param_val_list)}( d_fcs_data_by_event, d_clusters, num_dimensions, original_num_clusters, num_events);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Seed Kernel execution failed: ");
   
  // Computes the R matrix inverses, and the gaussian constant
  constants_kernel_launch${'_'+'_'.join(param_val_list)}(d_clusters,original_num_clusters,num_dimensions);
  cudaThreadSynchronize();
  CUT_CHECK_ERROR("Constants Kernel execution failed: ");
    
  // Calculate an epsilon value
  float epsilon = (1+num_dimensions+0.5*(num_dimensions+1)*num_dimensions)*log((float)num_events*num_dimensions)*0.0001;
  int iters;
  float likelihood, old_likelihood;
  // Used to hold the result from regroup kernel
  float* likelihoods = (float*) malloc(sizeof(float)*${num_blocks_estep});
  float* d_likelihoods;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*${num_blocks_estep}));
    
  // Variables for GMM reduce order
  float distance, min_distance = 0.0;
  int min_c1, min_c2;

  /*************** EM ALGORITHM *****************************/
        
  //================================== EM INITIALIZE =======================

  estep1_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_clusters, d_cluster_memberships, num_dimensions,num_events,d_likelihoods,num_clusters);
  //cudaThreadSynchronize();

  estep2_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_clusters, d_cluster_memberships, num_dimensions,num_clusters,num_events,d_likelihoods);
  cudaThreadSynchronize();

  // check if kernel execution generated an error
  CUT_CHECK_ERROR("Kernel execution failed");

  // Copy the likelihood totals from each block, sum them up to get a total
  CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*${num_blocks_estep},cudaMemcpyDeviceToHost));
  likelihood = 0.0;
  for(int i=0;i<${num_blocks_estep};i++) {
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
  while(iters < ${min_iters} || (iters < ${max_iters} && fabs(change) > epsilon)) {
    old_likelihood = likelihood;
            
    //params = M step
    // This kernel computes a new N, pi isn't updated until compute_constants though
    mstep_N_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_event,d_clusters, d_cluster_memberships, num_dimensions,num_clusters,num_events);
    cudaThreadSynchronize();

    // This kernel computes new means
    mstep_means_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_clusters, d_cluster_memberships, num_dimensions,num_clusters,num_events);
    cudaThreadSynchronize();
            
%if covar_version_name.upper() in ['2B','V2B','_V2B']:
      CUDA_SAFE_CALL(cudaMemcpy(temp_buffer_2b, zeroR_2b, sizeof(float)*num_dimensions*num_dimensions*num_clusters, cudaMemcpyHostToDevice) );
%endif

    // Covariance is symmetric, so we only need to compute N*(N+1)/2 matrix elements per cluster
    mstep_covar_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_fcs_data_by_event,d_clusters,d_cluster_memberships,num_dimensions,num_clusters,num_events,temp_buffer_2b);
    cudaThreadSynchronize();
                 
    CUT_CHECK_ERROR("M-step Kernel execution failed: ");


    // Inverts the R matrices, computes the constant, normalizes cluster probabilities
    constants_kernel_launch${'_'+'_'.join(param_val_list)}(d_clusters,num_clusters,num_dimensions);
    cudaThreadSynchronize();
    CUT_CHECK_ERROR("Constants Kernel execution failed: ");

    //regroup = E step
    // Compute new cluster membership probabilities for all the events
    estep1_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_clusters,d_cluster_memberships, num_dimensions,num_events,d_likelihoods,num_clusters);

    estep2_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_clusters,d_cluster_memberships, num_dimensions,num_clusters,num_events,d_likelihoods);
    cudaThreadSynchronize();

    CUT_CHECK_ERROR("E-step Kernel execution failed: ");
        
    // check if kernel execution generated an error
    CUT_CHECK_ERROR("Kernel execution failed");
        
    CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*${num_blocks_estep},cudaMemcpyDeviceToHost));
    likelihood = 0.0;
    for(int i=0;i<${num_blocks_estep};i++) {
      likelihood += likelihoods[i]; 
    }
            
    change = likelihood - old_likelihood;
    //printf("Iter %d likelihood = %f\n", iters, likelihood);
    //printf("Change in likelihood: %f (vs. %f)\n",change, epsilon);

    iters++;
    
  }//EM Loop
        

  //================================ EM DONE ==============================
  //printf("\nFinal rissanen Score was: %f, with %d clusters.\n",min_rissanen,num_clusters);
  //printf("DONE COMPUTING\n");

  copy_cluster_data_GPU_to_CPU(num_clusters, num_dimensions);
  
  // cleanup host memory
  free(likelihoods);
  free(cluster_memberships);

%if covar_version_name.upper() in ['2B','V2B','_V2B']:
//TODO: free these  
//zeroR_2b
//temp_buffer_2b
%endif

  // cleanup GPU memory
  cudaFree(d_likelihoods);
  cudaFree(d_cluster_memberships);
  
  return likelihood;

}

