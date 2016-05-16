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

#include <deal.II/lac/cuda_vector.h>
#include <deal.II/base/exceptions.h>

#ifdef DEAL_II_WITH_CUDA

DEAL_II_NAMESPACE_OPEN

namespace LinearAlgebra
{
  namespace CUDAWrappers
  {
    template <typename Number>
    Vector<Number>::Vector()
    :
      val(nullptr),
      n_elements(0)
    {}



    template <typename Number>
    Vector<Number>::Vector(const Vector<Number> &V)
    :
      n_elements(V.n_elements)
    {
      // Allocate the memory
      cudaError_t error_code = cudaMalloc(&val, n_elements*sizeof(Number));
      CudaAssert(error_code);
      // Copy the values.
      error_code = cudaMemcpy(val, V.val,n_elements*sizeof(Number),
          cudaMemcpyDeviceToDevice);
      CudaAssert(error_code);
    }



    template <typename Number>
    Vector(const size_type n)
    :
      n_elements(n)
    {
      // Allocate the memory
      cudaError_t error_code = cudaMalloc(&val, n_elements*sizeof(Number));
      CudaAssert(error_code);
    }



    template <typename Number>
    Vector<Number>::~Vector()
    {
      if (val != nullptr)
      {
        cudaError_t error_code = cudaFree(val);
        CudaAssert(error_code);
        val = nullptr;
        n_elements = 0;
      }
    }



    template <typename Number>
    Vector<Number>::reinit(const size_type n,
                           const bool      omit_zeroing_entries)
    {
      // Resize the underlying array if necessary
      if (n == 0)
      {
        if (val != nullptr)
        {
          cudaError_t error_code = cudaFree(val);
          CudaAssert(error_code);
          val = nullptr;
        }
      }
      else
      {
        if (n_element != n)
        {
          cudaError_t error_code = cudaFree(val);
          CudaAssert(error_code);
        }

        cudaError_t error_code = cudaMalloc(&val, n_elements*sizeof(Number));
        CudaAssert(error_code);
      }
      n_elements = n;

      // If necessary set the elements to zero
      if (omit_zeroing_entries == false)
      {
        cudaError_t error_code = cudaMemset(val, Number(),
            n_elements*sizeof(Number));
        CudaAssert(error_code);
      }
    }
  }
}

DEAL_II_NAMESPACE_CLOSE

#endif
