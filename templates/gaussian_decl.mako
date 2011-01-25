
void seed_clusters_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event, clusters_t* d_clusters, int num_dimensions, int original_num_clusters, int num_events);
void constants_kernel_launch${'_'+'_'.join(param_val_list)}(clusters_t* d_clusters, int original_num_clusters, int num_dimensions);
void estep1_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_events, float* d_likelihoods, int num_clusters);
void estep2_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events, float* d_likelihoods);
void mstep_N_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_event, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events);
void mstep_means_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events);
void mstep_covar_launch${'_'+'_'.join(param_val_list)}(float* d_fcs_data_by_dimension, float* d_fcs_data_by_event, clusters_t* d_clusters, float* cluster_memberships, int num_dimensions, int num_clusters, int num_events, float* temp_buffer_2b);
