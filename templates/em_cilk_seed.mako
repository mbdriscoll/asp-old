//TODO: the only reason this is not simply a helper function is because in the CUDA VERSION the kernel launched by seed_components_launch${}() uses average_variance${}(), a function specialized based on $max_dimension, and it is easier to make the same functions specialized the same way in both Cuda dn Cilk versions
void em_cilk_seed_components${'_'+'_'.join(param_val_list)}(int num_components, int num_dimensions, int num_events) {
    seed_components${'_'+'_'.join(param_val_list)}(fcs_data_by_event, &components, num_dimensions, num_components, num_events);
  for(int m = 0; m < num_components; m++){
        printf("%0.4f ", components.N[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
        printf("%0.4f ", components.pi[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
        printf("%0.4f ", components.CP[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
        printf("%0.4f ", components.constant[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
        printf("%0.4f ", components.avgvar[m]);
  } printf("\n");
  for(int m = 0; m < num_components; m++){
    for(int d = 0; d < num_dimensions; d++)
        printf("%0.4f ", components.means[m*num_dimensions+d]);
    printf("\n");
  }
    for(int m = 0; m < num_components; m++){
        for(int d = 0; d < num_dimensions; d++)
            for(int d2 = 0; d2 < num_dimensions; d2++)
                printf("%0.4f ",
components.R[m*num_dimensions*num_dimensions+d*num_dimensions+d2]);
        printf("\n");
    }   
        
    for(int m = 0; m < num_components; m++){                                                            
        for(int d = 0; d < num_dimensions; d++)
            for(int d2 = 0; d2 < num_dimensions; d2++)                                                  
                printf("%0.4f ",
components.Rinv[m*num_dimensions*num_dimensions+d*num_dimensions+d2]); 
        printf("\n");
    }

}
