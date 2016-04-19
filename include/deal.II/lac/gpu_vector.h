#ifndef dealii__gpu_vector_h
#define dealii__gpu_vector_h

#include <deal.II/base/config.h>

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

#endif
