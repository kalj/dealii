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



#include "../tests.h"
#include <deal.II/base/logstream.h>
#include <deal.II/lac/cuda_vector.h>
#include <cmath>
#include <fstream>
#include <iomanip>




const unsigned int N=10;


template <typename number>
void print (const CUDAWrappers::Vector<number> &v)
{

  v.print(deallog);
}



template <typename number1>
void check_vector (Vector<number1> &v1)
{
  deallog << "Fill & Swap" << std::endl;
  CUDAWrappers::Vector<number1> v2(v1.size());
  print (v2);

  // initialize all entries to value
  v2 = 1.95;

  print (v2);

  // assignment
  v1 = v2;
  print (v1);

  // copy from CPU
  Vector<number1> cpu_v(N);

  for (unsigned int i=0; i<N; ++i)
    cpu_v(i) = 2. * i;

  v1.import(cpu_v, VectorOperation::insert, nullptr);

  print(v1);

  // swap
  swap (v1, v2);

  print (v1);

  print (v2);


  // FIXME: Fill in code to test other functions, and/or copy from lac/vector-vector.cc
}


int main()
{
  std::ofstream logfile("output");
  deallog << std::fixed;
  deallog << std::setprecision(2);
  deallog.attach(logfile);
  deallog.threshold_double(1.e-10);

  CUDAWrappers::Vector<double>      v(N);
  // do the same for float...

  check_vector (v);
}
