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

#ifndef dealii__gpu_vector_h
#define dealii__gpu_vector_h

#include <deal.II/base/config.h>

#ifdef DEAL_II_WITH_CUDA

// includes...

DEAL_II_NAMESPACE_OPEN

class GpuVector {
private:
  int N;
  float *x;
public:
  GpuVector(int N);
  ~GpuVector();
  void foo();
  void hello();
};


DEAL_II_NAMESPACE_CLOSE

#endif    // DEAL_II_WITH_CUDA

#endif    // dealii__gpu_vector_h
