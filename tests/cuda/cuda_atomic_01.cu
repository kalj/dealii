
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

// Test operator[] and norm_square of cuda_tensor.

#include "../tests.h"
#include <deal.II/lac/cuda_atomic.cuh>
#include <deal.II/base/logstream.h>
#include <fstream>
#include <iomanip>

__global__ void init_kernel(float *f, int N)
{
  const unsigned int i = threadIdx.x+blockIdx.x*blockDim.x;
  if(i<N)
    f[i] = threadIdx.x;
}

__global__ void sum_kernel(float *f, int N)
{
  const unsigned int i = threadIdx.x+blockIdx.x*blockDim.x;
  if(i<N)
    LinearAlgebra::CUDAWrappers::atomicAdd_wrapper(&f[0], f[i]);
}



int main ()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(5);
  deallog.attach(logfile);
  deallog.threshold_double(1.e-10);

  const int N = 100;
  float *f_dev, *f_host;
  cudaMalloc(&f_dev,N*sizeof(float));
  f_host = new float[N];

  init_kernel<<<10,10>>>(f_dev,N);
  sum_kernel<<<10,10>>>(f_dev,N);

  cudaMemcpy(f_host,f_dev,N*sizeof(float),cudaMemcpyDeviceToHost);

  deallog.push("values");
  for (unsigned int j=0; j<N; ++j)
    deallog << f_host[j] << std::endl;

  cudaFree(f_dev);
  delete[] f_host;

}