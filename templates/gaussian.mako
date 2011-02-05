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

typedef struct return_cluster_container
{
  boost::python::object cluster;
  //pyublas::numpy_array<float> distance;
  float distance;
} ret_c_con_t;

ret_c_con_t ret;
  
//=== Data structure pointers ===

//CPU copies of events
float *fcs_data_by_event;
float *fcs_data_by_dimension;

//GPU copies of events
float* d_fcs_data_by_event;
float* d_fcs_data_by_dimension;

//CPU copies of clusters
clusters_t clusters;
clusters_t saved_clusters;
clusters_t** scratch_cluster_arr; // for computing distances and merging
static int num_scratch_clusters = 0;

//CPU copies of eval data
float *cluster_memberships;
float *loglikelihoods;

//GPU copies of clusters
clusters_t temp_clusters;
clusters_t* d_clusters;

//GPU copies of eval data
float *d_cluster_memberships;
float *d_loglikelihoods;

//=================================

//AHC functions
void copy_cluster(clusters_t *dest, int c_dest, clusters_t *src, int c_src, int num_dimensions);
void add_clusters(clusters_t *clusters, int c1, int c2, clusters_t *temp_cluster, int num_dimensions);
float cluster_distance(clusters_t *clusters, int c1, int c2, clusters_t *temp_cluster, int num_dimensions);
//end AHC functions

//Copy functions to ensure CPU data structures are up to date
void copy_cluster_data_GPU_to_CPU(int num_clusters, int num_dimensions);
void copy_evals_data_GPU_to_CPU(int num_events, int num_clusters);

// Function prototypes
void writeCluster(FILE* f, clusters_t clusters, int c,  int num_dimensions);
void printCluster(clusters_t clusters, int c, int num_dimensions);
void invert_cpu(float* data, int actualsize, float* log_determinant);
int invert_matrix(float* a, int n, float* determinant);

clusters_t* alloc_temp_cluster_on_CPU(int num_dimensions) {

  clusters_t* scratch_cluster = (clusters_t*)malloc(sizeof(clusters_t));

  scratch_cluster->N = (float*) malloc(sizeof(float));
  scratch_cluster->pi = (float*) malloc(sizeof(float));
  scratch_cluster->constant = (float*) malloc(sizeof(float));
  scratch_cluster->avgvar = (float*) malloc(sizeof(float));
  scratch_cluster->means = (float*) malloc(sizeof(float)*num_dimensions);
  scratch_cluster->R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);
  scratch_cluster->Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions);

  return scratch_cluster;
}

void dealloc_temp_clusters_on_CPU() {

for(int i = 0; i<num_scratch_clusters; i++) {
  free(scratch_cluster_arr[i]->N);
  free(scratch_cluster_arr[i]->pi);
  free(scratch_cluster_arr[i]->constant);
  free(scratch_cluster_arr[i]->avgvar);
  free(scratch_cluster_arr[i]->means);
  free(scratch_cluster_arr[i]->R);
  free(scratch_cluster_arr[i]->Rinv);
  }

  return;
}
// ================== Event data allocation on CPU  ================= :
void alloc_events_on_CPU(pyublas::numpy_array<float> input_data, int num_events, int num_dimensions) {

  //printf("Alloc events on CPU\n");

  fcs_data_by_event = input_data.data();
  // Transpose the event data (allows coalesced access pattern in E-step kernel)
  // This has consecutive values being from the same dimension of the data 
  // (num_dimensions by num_events matrix)
  fcs_data_by_dimension  = (float*) malloc(sizeof(float)*num_events*num_dimensions);

  for(int e=0; e<num_events; e++) {
    for(int d=0; d<num_dimensions; d++) {
      fcs_data_by_dimension[d*num_events+e] = fcs_data_by_event[e*num_dimensions+d];
    }
  }
  return;
}

// ================== Event data allocation on GPU  ================= :

void alloc_events_on_GPU(int num_dimensions, int num_events) {
  //printf("Alloc events on GPU\n");
  int mem_size = num_dimensions*num_events*sizeof(float);
    
  // allocate device memory for FCS data
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_event, mem_size));
  CUDA_SAFE_CALL(cudaMalloc( (void**) &d_fcs_data_by_dimension, mem_size));

  return;
}

//hack hack..
void relink_clusters_on_CPU(pyublas::numpy_array<float> weights, pyublas::numpy_array<float> means, pyublas::numpy_array<float> covars) {
     clusters.pi = weights.data();
     clusters.means = means.data();
     clusters.R = covars.data();
}

// ================== Cluster data allocation on CPU  ================= :

void alloc_clusters_on_CPU(int original_num_clusters, int num_dimensions, pyublas::numpy_array<float> weights, pyublas::numpy_array<float> means, pyublas::numpy_array<float> covars) {

  //printf("Alloc clusters on CPU\n");
  
  //clusters.pi = (float*) malloc(sizeof(float)*original_num_clusters);
  //clusters.means = (float*) malloc(sizeof(float)*num_dimensions*original_num_clusters);   
  //clusters.R = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
  clusters.pi = weights.data();
  clusters.means = means.data();
  clusters.R = covars.data();

  clusters.N = (float*) malloc(sizeof(float)*original_num_clusters);      
  clusters.constant = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.avgvar = (float*) malloc(sizeof(float)*original_num_clusters);
  clusters.Rinv = (float*) malloc(sizeof(float)*num_dimensions*num_dimensions*original_num_clusters);
 
  return;
}  

// ================== Cluster data allocation on GPU  ================= :
void alloc_clusters_on_GPU(int original_num_clusters, int num_dimensions) {

  //printf("Alloc clusters on GPU\n");

  // Setup the cluster data structures on device
  // First allocate structures on the host, CUDA malloc the arrays
  // Then CUDA malloc structures on the device and copy them over
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.N),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.pi),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.constant),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.avgvar),sizeof(float)*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.means),sizeof(float)*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.R),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
  CUDA_SAFE_CALL(cudaMalloc((void**) &(temp_clusters.Rinv),sizeof(float)*num_dimensions*num_dimensions*original_num_clusters));
   
  // Allocate a struct on the device 
  CUDA_SAFE_CALL(cudaMalloc((void**) &d_clusters, sizeof(clusters_t)));
    
  // Copy Cluster data to device
  CUDA_SAFE_CALL(cudaMemcpy(d_clusters,&temp_clusters,sizeof(clusters_t),cudaMemcpyHostToDevice));

  return;
}

// ================= Eval data alloc on CPU and GPU =============== 

void alloc_evals_on_CPU(pyublas::numpy_array<float> cluster_mem_np_arr){
  cluster_memberships = cluster_mem_np_arr.data();
}

void alloc_evals_on_GPU(int num_events, int num_clusters){
  CUDA_SAFE_CALL(cudaMalloc((void**) &(d_cluster_memberships),sizeof(float)*num_events*num_clusters));
}

// ======================== Copy event data from CPU to GPU ================
void copy_event_data_CPU_to_GPU(int num_events, int num_dimensions) {

  //printf("Copy events to GPU\n");
  int mem_size = num_dimensions*num_events*sizeof(float);
  // copy FCS to device
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_event, fcs_data_by_event, mem_size,cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL(cudaMemcpy( d_fcs_data_by_dimension, fcs_data_by_dimension, mem_size,cudaMemcpyHostToDevice) );
  return;
}

// ======================== Copy cluster data from CPU to GPU ================
void copy_cluster_data_CPU_to_GPU(int num_clusters, int num_dimensions) {

   CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.N, clusters.N, sizeof(float)*num_clusters,cudaMemcpyHostToDevice)); 
   CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.pi, clusters.pi, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.constant, clusters.constant, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.avgvar, clusters.avgvar, sizeof(float)*num_clusters,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.means, clusters.means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.R, clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(temp_clusters.Rinv, clusters.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyHostToDevice));
   CUDA_SAFE_CALL(cudaMemcpy(d_clusters,&temp_clusters,sizeof(clusters_t),cudaMemcpyHostToDevice));
   return;
}
// ======================== Copy cluster data from GPU to CPU ================
void copy_cluster_data_GPU_to_CPU(int num_clusters, int num_dimensions) {

  CUDA_SAFE_CALL(cudaMemcpy(&temp_clusters, d_clusters, sizeof(clusters_t),cudaMemcpyDeviceToHost));
  // copy all of the arrays from the structs
  CUDA_SAFE_CALL(cudaMemcpy(clusters.N, temp_clusters.N, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(clusters.pi, temp_clusters.pi, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(clusters.constant, temp_clusters.constant, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(clusters.avgvar, temp_clusters.avgvar, sizeof(float)*num_clusters,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(clusters.means, temp_clusters.means, sizeof(float)*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(clusters.R, temp_clusters.R, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(clusters.Rinv, temp_clusters.Rinv, sizeof(float)*num_dimensions*num_dimensions*num_clusters,cudaMemcpyDeviceToHost));
  
  return;
}

// ======================== Copy eval data from GPU to CPU ================
void copy_evals_data_GPU_to_CPU(int num_events, int num_clusters){
  CUDA_SAFE_CALL(cudaMemcpy(cluster_memberships, d_cluster_memberships, sizeof(float)*num_events*num_clusters, cudaMemcpyDeviceToHost));
}

// ================== Set the GPU Device ===================
void set_GPU_device(int device) {
  // Set the device to run on... 0 for GTX 480, 1 for GTX 285 on oak
  int GPUCount;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&GPUCount));
  if(GPUCount == 0) {
    //printf("Only 1 CUDA device found, defaulting to it.\n");
    device = 0;
  } else if (device >= GPUCount) {
    device  = GPUCount-1;
  }
  CUDA_SAFE_CALL(cudaSetDevice(device));
}


// ================== Event data dellocation on CPU  ================= :
void dealloc_events_on_CPU() {
  //free(fcs_data_by_event);
  free(fcs_data_by_dimension);
  return;
}

// ================== Event data dellocation on GPU  ================= :
void dealloc_events_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_event));
  CUDA_SAFE_CALL(cudaFree(d_fcs_data_by_dimension));
  return;
}


// ==================== Cluster data deallocation on CPU =================  
void dealloc_clusters_on_CPU() {

  //free(clusters.pi);
  //free(clusters.means);
  //free(clusters.R);

  free(clusters.N);
  free(clusters.constant);
  free(clusters.avgvar);
  free(clusters.Rinv);
  return;
}

// ==================== Cluster data deallocation on GPU =================  
void dealloc_clusters_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(temp_clusters.N));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.pi));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.constant));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.avgvar));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.means));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.R));
  CUDA_SAFE_CALL(cudaFree(temp_clusters.Rinv));
  
  CUDA_SAFE_CALL(cudaFree(d_clusters));

  return;
}

// ==================== Eval data deallocation on CPU and GPU =================  
void dealloc_evals_on_CPU() {
  //free(cluster_memberships);
  return;
}

void dealloc_evals_on_GPU() {
  CUDA_SAFE_CALL(cudaFree(d_cluster_memberships));
  return;
}

// Accessor functions for pi, means, covars 

pyublas::numpy_array<float> get_temp_cluster_pi(clusters_t* c){
  pyublas::numpy_array<float> ret = pyublas::numpy_array<float>(1);
  std::copy( c->pi, c->pi+1, ret.begin());
  return ret;
}

pyublas::numpy_array<float> get_temp_cluster_means(clusters_t* c, int D){
  pyublas::numpy_array<float> ret = pyublas::numpy_array<float>(D);
  std::copy( c->means, c->means+D, ret.begin());
  return ret;
}

pyublas::numpy_array<float> get_temp_cluster_covars(clusters_t* c, int D){
  pyublas::numpy_array<float> ret = pyublas::numpy_array<float>(D*D);
  std::copy( c->R, c->R+D*D, ret.begin());
  return ret;
}

//------------------------- AHC FUNCTIONS ----------------------------

int compute_distance_rissanen(int c1, int c2, int num_dimensions) {
  // compute distance function between the 2 clusters

  clusters_t *new_cluster = alloc_temp_cluster_on_CPU(num_dimensions);

  float distance = cluster_distance(&clusters,c1,c2,new_cluster,num_dimensions);
  //printf("distance %d-%d: %f\n", c1, c2, distance);

  scratch_cluster_arr[num_scratch_clusters] = new_cluster;
  num_scratch_clusters++;
  
  ret.cluster = boost::python::object(boost::python::ptr(new_cluster));
  ret.distance = distance;

  return 0;

}

void merge_clusters(int min_c1, int min_c2, clusters_t *min_cluster, int num_clusters, int num_dimensions) {

  // Copy new combined cluster into the main group of clusters, compact them
  copy_cluster(&clusters,min_c1, min_cluster,0,num_dimensions);

  for(int i=min_c2; i < num_clusters-1; i++) {
  
    copy_cluster(&clusters,i,&clusters,i+1,num_dimensions);
  }

  //return boost::python::object(boost::python::ptr(clusters));
  //return boost::python::object(clusters);
  return;
}


float cluster_distance(clusters_t *clusters, int c1, int c2, clusters_t *temp_cluster, int num_dimensions) {
  // Add the clusters together, this updates pi,means,R,N and stores in temp_cluster

  add_clusters(clusters,c1,c2,temp_cluster,num_dimensions);
  //printf("%f, %f, %f, %f, %f, %f\n", clusters->N[c1], clusters->constant[c1], clusters->N[c2], clusters->constant[c2], temp_cluster->N[0], temp_cluster->constant[0]);
  return clusters->N[c1]*clusters->constant[c1] + clusters->N[c2]*clusters->constant[c2] - temp_cluster->N[0]*temp_cluster->constant[0];
  
}

void add_clusters(clusters_t *clusters, int c1, int c2, clusters_t *temp_cluster, int num_dimensions) {
  float wt1,wt2;
 
  wt1 = (clusters->N[c1]) / (clusters->N[c1] + clusters->N[c2]);
  wt2 = 1.0f - wt1;
    
  // Compute new weighted means
  for(int i=0; i<num_dimensions;i++) {
    temp_cluster->means[i] = wt1*clusters->means[c1*num_dimensions+i] + wt2*clusters->means[c2*num_dimensions+i];
  }
    
  // Compute new weighted covariance
  for(int i=0; i<num_dimensions; i++) {
    for(int j=i; j<num_dimensions; j++) {
      // Compute R contribution from cluster1
      temp_cluster->R[i*num_dimensions+j] = ((temp_cluster->means[i]-clusters->means[c1*num_dimensions+i])
                                             *(temp_cluster->means[j]-clusters->means[c1*num_dimensions+j])
                                             +clusters->R[c1*num_dimensions*num_dimensions+i*num_dimensions+j])*wt1;
      // Add R contribution from cluster2
      temp_cluster->R[i*num_dimensions+j] += ((temp_cluster->means[i]-clusters->means[c2*num_dimensions+i])
                                              *(temp_cluster->means[j]-clusters->means[c2*num_dimensions+j])
                                              +clusters->R[c2*num_dimensions*num_dimensions+i*num_dimensions+j])*wt2;
      // Because its symmetric...
      temp_cluster->R[j*num_dimensions+i] = temp_cluster->R[i*num_dimensions+j];
    }
  }
    
  // Compute pi
  temp_cluster->pi[0] = clusters->pi[c1] + clusters->pi[c2];
    
  // compute N
  temp_cluster->N[0] = clusters->N[c1] + clusters->N[c2];

  float log_determinant;
  // Copy R to Rinv matrix
  memcpy(temp_cluster->Rinv,temp_cluster->R,sizeof(float)*num_dimensions*num_dimensions);
  // Invert the matrix
  invert_cpu(temp_cluster->Rinv,num_dimensions,&log_determinant);
  // Compute the constant
  temp_cluster->constant[0] = (-num_dimensions)*0.5*logf(2*PI)-0.5*log_determinant;
    
  // avgvar same for all clusters
  temp_cluster->avgvar[0] = clusters->avgvar[0];
}

void copy_cluster(clusters_t *dest, int c_dest, clusters_t *src, int c_src, int num_dimensions) {
  dest->N[c_dest] = src->N[c_src];
  dest->pi[c_dest] = src->pi[c_src];
  dest->constant[c_dest] = src->constant[c_src];
  dest->avgvar[c_dest] = src->avgvar[c_src];
  memcpy(&(dest->means[c_dest*num_dimensions]),&(src->means[c_src*num_dimensions]),sizeof(float)*num_dimensions);
  memcpy(&(dest->R[c_dest*num_dimensions*num_dimensions]),&(src->R[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  memcpy(&(dest->Rinv[c_dest*num_dimensions*num_dimensions]),&(src->Rinv[c_src*num_dimensions*num_dimensions]),sizeof(float)*num_dimensions*num_dimensions);
  // do we need to copy memberships?
}
//---------------- END AHC FUNCTIONS ----------------


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
  int i,imax=0,j,k;
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

