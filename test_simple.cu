#include<iostream>

#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include<map>
#include"cg_sparse.hpp"

template<typename T>
void print_vector(std::vector<T> v){

  for(int i=0; i<v.size(); ++i)
   std::cerr<<v[i]<<"\t";
  std::cerr<<std::endl;

}

#include"GetMatrixMarket.hpp"

int main(){

  inizialize_device_constant();

  std::vector<int> ii, jj;
  std::vector<double> data;
  int r,c,nnz;

  GetMatrixMarket_CSR_symm("494_bus.mtx", data, ii, jj, nnz, r, c);

  std::cerr<<"SPARSE MATRIX LOADED "<<std::endl;
 
 
  thrust::host_vector<double> x(r,1), b(r,1);


  thrust::device_vector<double> d_csrValA=data;
  thrust::device_vector<int> d_csrRowPtrA=ii;
  thrust::device_vector<int> d_csrColIndA=jj;


  thrust::device_vector<double> d_b=b;
  thrust::device_vector<double> d_x=x;
  double *pntr_d_csrValA = thrust::raw_pointer_cast(d_csrValA.data());
  int *pntr_d_csrRowPtrA = thrust::raw_pointer_cast(d_csrRowPtrA.data());
  int *pntr_d_csrColIndA = thrust::raw_pointer_cast(d_csrColIndA.data());
  double *pntr_d_b = thrust::raw_pointer_cast(d_b.data());
  double *pntr_d_x = thrust::raw_pointer_cast(d_x.data());

  std::cerr<<"# of iterations: "<<cg_sparse(NULL, pntr_d_csrValA, pntr_d_csrRowPtrA, pntr_d_csrColIndA, pntr_d_b, pntr_d_x, c, nnz);
  std::cerr<<std::endl<<"results: "<<std::endl;

  x=d_x;
  for(size_t i=0; i<r; ++i)
   std::cout<<x[i]<<"\t";
  std::cout<<std::endl;

  return 0;

} 
