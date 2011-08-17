
void compute_CP${'_'+'_'.join(param_val_list)}(components_t* components, int M, int D) {
    int row, col;
    cilk_for( int m = 0; m < M; m++) {
        cilk::reducer_opadd<float> total(0.0f);
        cilk_for(int d = 0; d < D; d++) {
            row = (d) / D; 
            col = (d) % D; 
            if(row==col) {
                total += logf(2*PI*components->R[m*D*D + row*D +col]);
            }
        }
        components->CP[m] = total.get_value();
    }
}

void average_variance${'_'+'_'.join(param_val_list)}(float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar) {
    
    cilk::reducer_opadd<float> total(0.0f);
    // Compute average variance for each dimension
    cilk_for(int i = 0; i < num_dimensions; i++) {
        float variance = 0.0f;
        for(int j=0; j < num_events; j++) {
            variance += fcs_data[j*num_dimensions + i]*fcs_data[j*num_dimensions + i];
        }
        variance /= (float) num_events;
        variance -= means[i]*means[i];
        total += variance;
    }
    
    *avgvar = total.get_value() / (float) num_dimensions;
}

void seed_covars${'_'+'_'.join(param_val_list)}(components_t* components, float* fcs_data, float* means, int num_dimensions, int num_events, float* avgvar, int num_components) {

    cilk_for(int i=0; i < num_dimensions*num_dimensions; i++) {
      int row = (i) / num_dimensions;
      int col = (i) % num_dimensions;
      components->R[row*num_dimensions+col] = 0.0f;
      for(int j=0; j < num_events; j++) {
        if(row==col) {
          components->R[row*num_dimensions+col] += (fcs_data[j*num_dimensions + row])*(fcs_data[j*num_dimensions + row]);
        }
      }
      if(row==col) {
        components->R[row*num_dimensions+col] /= (float) (num_events -1);
        components->R[row*num_dimensions+col] -= ((float)(num_events)*means[row]*means[row]) / (float)(num_events-1);
        components->R[row*num_dimensions+col] /= (float)num_components;
      }
    }
}

void seed_components${'_'+'_'.join(param_val_list)}(float *data, components_t* components, int num_dimensions, int num_components, int num_events) {
    float* means = (float*) malloc(sizeof(float)*num_dimensions);
    float avgvar;

    // Compute means
    for(int d=0; d < num_dimensions; d++) {
        means[d] = 0.0;
        for(int n=0; n < num_events; n++) {
            means[d] += data[n*num_dimensions+d];
        }
        means[d] /= (float) num_events;
    }

    // Compute the average variance
    seed_covars${'_'+'_'.join(param_val_list)}(components, data, means, num_dimensions, num_events, &avgvar, num_components);
    average_variance${'_'+'_'.join(param_val_list)}(data, means, num_dimensions, num_events, &avgvar);    
    float seed;
    if(num_components > 1) {
       seed = (num_events)/(num_components);
    } else {
       seed = 0.0f;
    }

    memcpy(components->means, means, sizeof(float)*num_dimensions);

    for(int c=1; c < num_components; c++) {
        memcpy(&components->means[c*num_dimensions], &data[((int)(c*seed))*num_dimensions], sizeof(float)*num_dimensions);
          
        for(int i=0; i < num_dimensions*num_dimensions; i++) {
          components->R[c*num_dimensions*num_dimensions+i] = components->R[i];
          components->Rinv[c*num_dimensions*num_dimensions+i] = 0.0f;
        }
    }

    //compute pi, N
    for(int c =0; c<num_components; c++) {
        components->pi[c] = 1.0f/((float)num_components);
        components->N[c] = ((float) num_events) / ((float)num_components);
        components->avgvar[c] = avgvar / COVARIANCE_DYNAMIC_RANGE;
    }

    free(means);
    compute_CP${'_'+'_'.join(param_val_list)}(components, num_components, num_dimensions);
}

void constants${'_'+'_'.join(param_val_list)}(components_t* components, int M, int D) {
    float log_determinant;
    float* matrix = (float*) malloc(sizeof(float)*D*D);

    float sum = 0.0;
    for(int m=0; m < M; m++) {
        // Invert covariance matrix
        memcpy(matrix,&(components->R[m*D*D]),sizeof(float)*D*D);
        invert_cpu(matrix,D,&log_determinant);
        memcpy(&(components->Rinv[m*D*D]),matrix,sizeof(float)*D*D);
    
        // Compute constant
        components->constant[m] = -D*0.5f*logf(2.0f*PI) - 0.5f*log_determinant;

        // Sum for calculating pi values
        //sum += components->N[m];
    }

    // Compute pi values
    //for(int m=0; m < M; m++) {
    //    components->pi[m] = components->N[m] / sum;
    //}
    
    free(matrix);
}

void estep1${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* likelihood, float* loglikelihoods) {
    // Compute likelihood for every data point in each component
    float* temploglikelihoods = (float*)malloc(M*N*sizeof(float));
    cilk_for(int m=0; m < M; m++) {
        float component_pi = components->pi[m];
        float constant = components->constant[m];
        float component_CP = components->CP[m];
        cilk_for(int n=0; n < N; n++) {
            float* means = (float*) &(components->means[m*D]);
            float* Rinv = (float*) &(components->Rinv[m*D*D]);
            float loglike = 0.0;
            #if ${diag_only}
            for(int i=0; i < D; i++) {
                loglike += (data[i+n*D]-means[i])*(data[i+n*D]-means[i])*Rinv[i*D+i];
            }
            #else
            for(int i=0; i < D; i++) {
                for(int j=0; j < D; j++) {
                    loglike += (data[i+n*D]-means[i])*(data[j+n*D]-means[j])*Rinv[i*D+j];
                }
            }
            #endif  
            loglike = (component_pi > 0.0f) ? -0.5*(loglike + component_CP) + logf(component_pi) : MINVALUEFORMINUSLOG;
            temploglikelihoods[m*N+n] = loglike;
            component_memberships[m*N+n] = -0.5f * loglike + component_constant + log(component_pi); 
        }
    }
    //estep1 log_add()
    cilk_for(int n=0; n < N; n++) {
        float finalloglike = MINVALUEFROMMINUSLOG;
        for(int m=0; m < M; m++) {
            finalloglike = log_add(finalloglike, temploglikelihoods[m*N+n]);
        }
        loglikelihoods[n] = finalloglike;s
    }
    free(temploglikelihoods);
}

float estep2_events${'_'+'_'.join(param_val_list)}(components_t* components, float* component_memberships, int M, int n, int N) {
	// Finding maximum likelihood for this data point
	float max_likelihood;
	max_likelihood = __sec_reduce_max(component_memberships[n:M:N]);

	// Computes sum of all likelihoods for this event
	float denominator_sum;
	denominator_sum = 0.0f;
	for(int m=0; m < M; m++) {
		denominator_sum += exp(component_memberships[m*N+n] - max_likelihood);
	}
	denominator_sum = max_likelihood + log(denominator_sum);

	// Divide by denominator to get each membership
	for(int m=0; m < M; m++) {
		component_memberships[m*N+n] = exp(component_memberships[m*N+n] - denominator_sum);
	}
        //or component_memberships[n:M:N] = exp(component_memberships[n:M:N] - denominator_sum);

	return denominator_sum;
}

void estep2${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N, float* likelihood) {
    cilk::reducer_opadd<float> total(0.0f);
    cilk_for(int n=0; n < N; n++) {
        total += estep2_events${'_'+'_'.join(param_val_list)}(components, component_memberships, M, n, N);
    }
    *likelihood = total.get_value();
}

void mstep_n${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N) {
    for(int m=0; m < M; m++) {
        components->N[m] = 0.0;
        for(int n=0; n < N; n++) {
            components->N[m] += component_memberships[m*N+n];
        }
    }
}

void mstep_mean${'_'+'_'.join(param_val_list)}(float* data, components_t* components, float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        for(int d=0; d < D; d++) {
	    components->means[m*D+d] = 0.0;
	    for(int n=0; n < N; n++) {
		components->means[m*D+d] += data[d*N+n]*component_memberships[m*N+n];
	    }
	    components->means[m*D+d] /= components->N[m];
        }
    }
}

void mstep_covar${'_'+'_'.join(param_val_list)}(float* data, components_t* components,float* component_memberships, int D, int M, int N) {
    cilk_for(int m=0; m < M; m++) {
        float* means = &(components->means[m*D]);
        float sum;
        for(int i=0; i < D; i++) {
            for(int j=0; j <= i; j++) {
                #if ${diag_only}
                if(i != j) {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                    continue;
                }
                #endif
                sum = 0.0;
                for(int n=0; n < N; n++) {
                    sum += (data[i*N+n]-means[i])*(data[j*N+n]-means[j])*component_memberships[m*N+n];
                }
                if(components->N[m] >= 1.0f) {
                    components->R[m*D*D+i*D+j] = sum / components->N[m];
                    components->R[m*D*D+j*D+i] = sum / components->N[m];
                } else {
                    components->R[m*D*D+i*D+j] = 0.0f;
                    components->R[m*D*D+j*D+i] = 0.0f;
                }
            }
        }
    }
    //TODO: Doing this because mstep_covar_launch${} in the Cuda version does it...not sure why
    compute_CP${'_'+'_'.join(param_val_list)}(components, M, D);
}
