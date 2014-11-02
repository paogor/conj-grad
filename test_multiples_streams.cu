#include<iostream>
#include"cg_sparse.hpp"
#include<thrust/host_vector.h>
#include<thrust/device_vector.h>
#include"GetMatrixMarket.hpp"
#include"CUDATIMER.hpp"

#define cudaDeviceScheduleBlockingSync 0x04

int main(int argc, const char* argv[]){

  int sn= 1;

  if ( argc > 1 ) {
    sn = std::atoi( argv[1] );
  }

  std::cout<<sn<<std::endl;

  inizialize_device_constant();

  std::vector<int> ii, jj;
  std::vector<double> data;
  int r,c,nnz;

  GetMatrixMarket_CSR_symm("494_bus.mtx", data, ii, jj, nnz, r, c);

  std::cerr<<"SPARSE MATRIX LOADED"<<r<<std::endl;
  
  thrust::host_vector<double> x(r,1), b(r,1);


  thrust::device_vector<double> d_csrValA=data;
  thrust::device_vector<int> d_csrRowPtrA=ii;
  thrust::device_vector<int> d_csrColIndA=jj;


  thrust::device_vector<double> d_b=b;
  double *pntr_d_csrValA = thrust::raw_pointer_cast(d_csrValA.data());
  int *pntr_d_csrRowPtrA = thrust::raw_pointer_cast(d_csrRowPtrA.data());
  int *pntr_d_csrColIndA = thrust::raw_pointer_cast(d_csrColIndA.data());
  double *pntr_d_b = thrust::raw_pointer_cast(d_b.data());

  const size_t streams_number = sn;

  std::vector<cudaStream_t> streamNo(streams_number);
  std::vector<thrust::device_vector<double> > x_vect(streams_number);

  for(size_t i = 0; i < streams_number; ++i){
    cudaStreamCreate(&streamNo[i]);
    x_vect[i]=x;
  }

  {

  #pragma omp parallel for num_threads(streams_number)
  for(size_t i = 0; i < streams_number; ++i){
   cg_sparse(streamNo[0], pntr_d_csrValA, pntr_d_csrRowPtrA, pntr_d_csrColIndA, pntr_d_b, thrust::raw_pointer_cast(x_vect[0].data()), r, nnz);
  }

    checkError(cudaDeviceSynchronize());

  }

  for(size_t i = 0; i < streams_number; ++i)
    cudaStreamDestroy(streamNo[i]);  	

/*
  x=x_vect[0];
  for(size_t i=0; i<r; ++i)
   std::cerr<<x[i]<<"\t";
  std::cerr<<std::endl;
*/

  return 0;

} 
