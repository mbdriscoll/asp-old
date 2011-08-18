//TODO: the only reason this is not simply a helper function is because in the CUDA VERSION the kernel launched by seed_components_launch${}() uses average_variance${}(), a function specialized based on $max_dimension, and it is easier to make the same functions specialized the same way in both Cuda dn Cilk versions
void em_cilk_seed_components${'_'+'_'.join(param_val_list)}(int num_components, int num_dimensions, int num_events) {
    seed_components${'_'+'_'.join(param_val_list)}(fcs_data_by_event, &components, num_dimensions, num_components, num_events);
}
