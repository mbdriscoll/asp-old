
// ================== Seed Components function - to initialize the clusters  ================= :

//TODO: the only reason this is not simply a helper function is because the kernel launched by seed_components_launch${}() uses average_variance${}(), a function specialized based on $max_dimension.
//	Therefore, all the functions in the call stack must also be specialized...
//TODO: do we want events be passed from Python?
void em_cuda_seed_components${'_'+'_'.join(param_val_list)}(int num_dimensions, int num_components, int num_events) {
  seed_components_launch${'_'+'_'.join(param_val_list)}(d_fcs_data_by_event, d_components, num_dimensions, num_components, num_events);
}

