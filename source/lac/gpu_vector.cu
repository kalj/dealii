#include <deal.II/lac/gpu_vector.h>
#include <cstdio>

DEAL_II_NAMESPACE_OPEN

__global__ void foo_kernel(float *x, int N)
{
  const unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if(idx<N)
    x[idx] = 1.3;
}


GpuVector::GpuVector(int N)
{
  this->N=N;
  cudaMalloc(&x,sizeof(float)*N);
}

GpuVector::~GpuVector()
{
  cudaFree(x);
}

void GpuVector::foo()
{
  dim3 bk_dim(128);
  dim3 gd_dim(1+(N-1)/128);
  foo_kernel<<<gd_dim,bk_dim>>>(this->x,this->N);
}

void GpuVector::hello()
{
  float x_host;
  cudaMemcpy(&x_host,x,sizeof(float),cudaMemcpyDeviceToHost);
  printf("x[0]=%g\n",x_host);
}

DEAL_II_NAMESPACE_CLOSE
