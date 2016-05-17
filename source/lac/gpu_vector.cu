// ---------------------------------------------------------------------
//
// Copyright (C) 2016 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

#include <deal.II/lac/gpu_vector.h>

#ifdef DEAL_II_WITH_CUDA

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

float GpuVector::hello()
{
  float x_host;
  cudaMemcpy(&x_host,x,sizeof(float),cudaMemcpyDeviceToHost);
  return x_host;
}

DEAL_II_NAMESPACE_CLOSE

#endif
