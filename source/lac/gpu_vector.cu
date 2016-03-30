


DEAL_II_NAMESPACE_OPEN


__global__ void foo_kernel(float *x)
{
  x[0] = 0;
}


GpuVector::GpuVector()
{

}
void GpuVector::foo()
{
  float *x;

  foo_kernel<<<1,1>>> foo_kernel(x);
}

DEAL_II_NAMESPACE_CLOSE