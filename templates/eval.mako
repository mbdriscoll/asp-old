
float eval${'_'+'_'.join(param_val_list)} (
                             int num_clusters, 
                             int num_dimensions, 
                             int num_events, 
                             pyublas::numpy_array<float> obs_data ) 
{
  float likelihood;
  // Used to hold the result from regroup kernel
  float* likelihoods = (float*) malloc(sizeof(float)*${num_blocks_estep});
  float* d_likelihoods;
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_likelihoods, sizeof(float)*${num_blocks_estep}));

  estep1_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_clusters, d_cluster_memberships, num_dimensions,num_events,d_likelihoods,num_clusters);
  //cudaThreadSynchronize();

  estep2_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_dimension,d_clusters, d_cluster_memberships, num_dimensions,num_clusters,num_events,d_likelihoods);
  cudaThreadSynchronize();

  CUT_CHECK_ERROR("Kernel execution failed");

  // Copy the likelihood totals from each block, sum them up to get a total
  CUDA_SAFE_CALL(cudaMemcpy(likelihoods,d_likelihoods,sizeof(float)*${num_blocks_estep},cudaMemcpyDeviceToHost));
  likelihood = 0.0;
  for(int i=0;i<${num_blocks_estep};i++) {
    likelihood += likelihoods[i]; 
  }

  copy_evals_data_GPU_to_CPU(num_events, num_clusters);

  return likelihood; 

}
