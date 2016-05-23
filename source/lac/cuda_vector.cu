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

#define BKSIZE_ELEMWISE_OP 512
#define CHUNKSIZE_ELEMWISE_OP 8
#define VR_BKSIZE 512
#define VR_CHUNK_SIZE 8
      
namespace LinearAlgebra
{
  namespace CUDAWrappers
  {
    namespace internal
    {
      template <typename Number>
      __global__ void vec_scale(Number *val, const Number a, const unsigned int N)
      {
        const unsigned int idx_base = threadIdx.x + blockIdx.x *
          (blockDim.x*CHUNKSIZE_ELEMWISE_OP);
        for (unsigned int i=0; i<CHUNKSIZE_ELEMWISE_OP; ++i)
        {
          const unsigned int idx = idx_base + i*BKSIZE_ELEMWISE_OP;
          if (idx<N)
            val[idx] = a;
        }
      }



      struct Binop_Addition
      {
        template <typename Number>
        __device__ static inline Number operation(const Number a, 
            const Number b)
        {
          return a+b;
        }
      };



      struct Binop_Subtraction
      {
        template <typename Number>
        __device__ static inline Number operation(const Number a,
            const Number b)
        {
          return a-b;
        }
      };



      template <typename Number, typename Binop>
      __global__ void vector_bin_op(Number* v1, Number* v2, const int N)
      {
        const unsigned int idx_base = threadIdx.x + blockIdx.x *
          (blockDim.x*CHUNKSIZE_ELEMWISE_OP);
        for (unsigned int i=0; i<CHUNKSIZE_ELEMWISE_OP; ++i)
        {
          const unsigned int idx = idx_base + i*BKSIZE_ELEMWISE_OP;
          if (idx<N)
            v1[idx] = Binop::operation(v1[idx],v2[idx]);
        }
      }



      template <typename Number, typename Operation>
      __device__ void reduce_within_warp(volatile Number* result_buffer,
          unsigned int local_idx)
      {
        if (VR_BKSIZE >= 64)
          result_buffer[local_idx] = 
            Operation::reduction_op(result_buffer[local_idx],
                result_buffer[local_idx+32]);
        if (VR_BKSIZE >= 32)
          result_buffer[local_idx] = 
            Operation::reduction_op(result_buffer[local_idx],
                result_buffer[local_idx+16]);
        if (VR_BKSIZE >= 16)
          result_buffer[local_idx] = 
            Operation::reduction_op(result_buffer[local_idx],
                result_buffer[local_idx+8]);
        if (VR_BKSIZE >= 8)
          result_buffer[local_idx] = 
            Operation::reduction_op(result_buffer[local_idx],
                result_buffer[local_idx+4]);
        if (VR_BKSIZE >= 4)
          result_buffer[local_idx] = 
            Operation::reduction_op(result_buffer[local_idx],
                result_buffer[local_idx+2]);
        if (VR_BKSIZE >= 2)
          result_buffer[local_idx] = 
            Operation::reduction_op(result_buffer[local_idx],
                result_buffer[local_idx+1]);
      }



      template <typename Number, typename Operation>
      __device__ void reduce(Number* result, Number* result_buffer,
          const unsigned int local_idx, const unsigned int global_idx, 
          const unsigned int N)
      {
        // TODO why 32?
        for (unsigned int s=VR_BKSIZE/2; s>32; s=s>>1)
        {
          if (local_idx < s)
            result_buffer[local_idx] = Operation::reduction_op(result_buffer[local_idx],
                  result_buffer[local_idx+s]);
          __syncthreads();
        }

        if (local_idx < 32)
          reduce_within_warp<Number,Operation>(result_buffer, local_idx);

        if (local_idx == 0)
          Operation::atomic_op(result, result_buffer[0]);
      }



      template <typename Number>
      struct DotProduct
      {
        __device__ static Number binary_op(const Number a, const Number b) 
        {
          return a*b;
        }

        __device__ static Number reduction_op(const Number a, const Number b)
        {
          return a+b;
        }

        __device__ static Number atomic_op(const Number a, Number* dst)
        {
          return atomicAdd(dst, a);
        }

        __device__ static Number null_value()
        {
          return Number();
        }
      };



      template <typename Number, typename Operation>
      __global__ void double_vector_reduction(Number *result, const Number* v1,
          const Number v2, const int N)
      {
        __shared__ Number result_buffer[VR_BKSIZE];

        const unsigned int global_idx = threadIdx.x + blockIdx.x*(blockDim.x*VR_CHUNK_SIZE);
        const unsigned int local_idx = threadIdx.x;

        if (global_idx<N)
          result_buffer[local_idx] = Operation::binary_op(v1[global_idx],v2[global_idx]);
        else
          result_buffer[local_idx] = Operation::null_value();

        for (unsigned int i=1; i<VR_CHUNK_SIZE; ++i)
        {
          const unsigned int idx = global_idx + i*VR_BKSIZE;
          if (idx<N)
            result_buffer[local_idx] =
              Operation::reduction_op(result_buffer[local_idx],
                  Operation::binary_op(v1[idx], v2[idx]));
        }

        __syncthreads();

        reduce<Number,Operation> (result,result_buffer,local_idx,global_idx,N);
      }
    }



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
    Vector<Number>::Vector(const size_type n)
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
    void Vector<Number>::reinit(const size_type n,
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
        if (n_elements != n)
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



    template <typename Number>
    void Vector<Number>::import(const ReadWriteVector<Number> &V,
        VectorOperation::values operation,
        std_cxx11::shared_ptr<const CommunicationPatternBase> )
    {
      //TODO
    }



    template <typename Number>
    Vector<Number>& Vector<Number>::operator*= (const Number factor)
    {
      AssertIsFinite(factor);
      const int n_blocks = 1 +
        (n_elements-1)/(CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      internal::vec_scale<Number> <<<n_blocks,BKSIZE_ELEMWISE_OP>>>(val,
          factor, n_elements);

      // Check that the kernel was launched correctly
      CudaAssert(cudaGetLastError());
      // Check that there was no problem during the execution of the kernel
      CudaAssert(cudaDeviceSynchronize());

      return *this;
    }



    template <typename Number>
    Vector<Number>& Vector<Number>::operator/= (const Number factor)
    {
      AssertIsFinite(factor);
      Assert(factor!=Number(0.), ExcZero());
      const int n_blocks = 1 +
        (n_elements-1)/(CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);
      internal::vec_scale<Number> <<<n_blocks,BKSIZE_ELEMWISE_OP>>>(val,
          1./factor, n_elements);

      // Check that the kernel was launched correctly
      CudaAssert(cudaGetLastError());
      // Check that there was no problem during the execution of the kernel
      CudaAssert(cudaDeviceSynchronize());

      return *this;
    }



    template <typename Number>
    Vector<Number>& Vector<Number>::operator+= (const VectorSpaceVector<Number> &V)
    {
      // Check that casting will work
      Assert(dynamic_cast<const Vector<Number>*>(&V)!=nullptr,
          ExcVectorTypeNotCompatible());

      // Downcast V. If fails, throws an exception.
      const Vector<Number> &down_V = dynamic_cast<const Vector<Number>&>(V);
      Assert(down_V.size()==this->Size(),
          ExcMessage("Cannot add two vectors with different numbers of elements"));

      const int n_blocks = 1 +
        (n_elements-1)/(CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);

      internal::vector_bin_op<Number,internal::Binop_Addition>
        <<<n_blocks,BKSIZE_ELEMWISE_OP>>>(val, down_V.val, n_elements);

      // Check that the kernel was launched correctly
      CudaAssert(cudaGetLastError());
      // Check that there was no problem during the execution of the kernel
      CudaAssert(cudaDeviceSynchronize());

      return *this;
    }



    template <typename Number>
    Vector<Number>& Vector<Number>::operator-= (const VectorSpaceVector<Number> &V)
    {
      // Check that casting will work
      Assert(dynamic_cast<const Vector<Number>*>(&V)!=nullptr,
          ExcVectorTypeNotCompatible());

      // Downcast V. If fails, throws an exception.
      const Vector<Number> &down_V = dynamic_cast<const Vector<Number>&>(V);
      Assert(down_V.size()==this->Size(),
          ExcMessage("Cannot add two vectors with different numbers of elements"));

      const int n_blocks = 1 +
        (n_elements-1)/(CHUNKSIZE_ELEMWISE_OP*BKSIZE_ELEMWISE_OP);

      internal::vector_bin_op<Number,internal::Binop_Subtraction>
        <<<n_blocks,BKSIZE_ELEMWISE_OP>>>(val, down_V.val, n_elements);

      // Check that the kernel was launched correctly
      CudaAssert(cudaGetLastError());
      // Check that there was no problem during the execution of the kernel
      CudaAssert(cudaDeviceSynchronize());

      return *this;
    }



    template <typename Number>
    Number Vector<Number>::operator* (const VectorSpaceVector<Number> &V)
    {
      // Check that casting will work
      Assert(dynamic_cast<const Vector<Number>*>(&V)!=nullptr,
          ExcVectorTypeNotCompatible());

      // Downcast V. If fails, throws an exception.
      const Vector<Number> &down_V = dynamic_cast<const Vector<Number>&>(V);
      Assert(down_V.size()==this->Size(),
          ExcMessage("Cannot add two vectors with different numbers of elements"));

      Number* result_device;
      cudaError_t error_code = cudaMalloc(&result_device, n_elements*sizeof(Number));
      CudaAssert(error_code);
      error_code = cudaMemset(result_device, Number(), sizeof(Number));

      const int n_blocks = 1 + (n_elements-1)/(VR_CHUNK_SIZE*VR_BKSIZE);
      internal::double_vector_reduction<internal::DotProduct<Number>>
        <<<dim3(n_blocks,1),dim3(VR_BKSIZE)>>> (result_device, val,
            down_V.val, n_elements);

      // Copy the result back to the host
      Number result;
      error_code = cudaMemcpy(&result, result_device, sizeof(Number),
          cudaMemcpyDeviceToHost);
      CudaAssert(error_code);
      // Free the memory on the device
      error_code = cudaFree(result_device);
      CudaAssert(error_code);

      return result;
    }
  }
}

DEAL_II_NAMESPACE_CLOSE

#endif
