#ifndef __CG_SPARSE_HPP__
#define __CG_SPARSE_HPP__

#include<cublas_v2.h> 
#include<cusparse_v2.h>
#include"CUDA_ERRORS.hpp"


/* A b x already present on GPU */
/* A need to be symmetric and positive definite */

/* device constant passed to cublas e cusparse functions */
__constant__ double _one_;
__constant__ double _minus_one_;
__constant__ double _zero_;

void inizialize_device_constant(){
// think about the case of multiple devices
  double _zero =.0, _one = 1., _minus_one = -1.;
  checkError(cudaMemcpyToSymbol(_zero_, &_zero,sizeof(double)));
  checkError(cudaMemcpyToSymbol(_one_, &_one,sizeof(double)));
  checkError(cudaMemcpyToSymbol(_minus_one_, &_minus_one,sizeof(double)));

}


/*
 * alpha_kernel   and   beta_kernel 
 * avoid the overhead of multiple call of cublas routines 
 *
 */

__global__ void alpha_kernel(int n, double *x, double *r, double *p, double *Ap, double *rr_old, double *pAp){

  __shared__ double alpha;
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if (threadIdx.x == 0 ) alpha = *rr_old / *pAp;

  __syncthreads();

  if(idx<n){
    // compute x
    x[idx] = x[idx] + alpha*p[idx];
    // compute r
    r[idx] = r[idx] - alpha*Ap[idx];
  }  


}

__global__ void beta_kernel(int n, double *p, double *r, double *rr_new, double *rr_old){

  __shared__ double beta;
  int idx = blockDim.x*blockIdx.x + threadIdx.x;
  if(threadIdx.x == 0 ) beta = *rr_new / *rr_old;

  __syncthreads();

  if(idx<n)
    p[idx] = r[idx] + beta*p[idx];

  if(idx==0) *rr_old = *rr_new;

}

/**
 * \fn cg_sparse
 * \brief Conjugate Gradient function
 * 
 * Solve linear system Ax=b
 *
 * A    system matrix in CSR format
 * \param b    rhs
 * \param x    solution 
 * \param n    dimension of the system
 * \param nnz  number of non-zero elements in A
 *
 */
int cg_sparse( cudaStream_t stream,
               const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
               double *b, double *x, const int n, const int nnz, double tol=1e-9 )
{

  /* ------------------------------------------------------------------------------------------------ */

  /* dimensions for kernel execution (tune for performances) */
  const int tpb = 64;
  const int nob = (n+tpb-1)/tpb;

  /* maximum iterations */
  const unsigned int kmax = 100000; 

  /* ------------------------------------------------------------------------------------------------ */

  /* get constants addresses inizialized in main */
  double *zero, *one, *minus_one;
  checkError(cudaGetSymbolAddress((void**)&one, _one_));
  checkError(cudaGetSymbolAddress((void**)&minus_one, _minus_one_));
  checkError(cudaGetSymbolAddress((void**)&zero, _zero_));

  /* allocating space for vectors on device (need to be freed at the end) */
  double *r, *z, *t, *p, *Ap; 
  checkError(cudaMalloc((void**)&p, n*sizeof(double))); 
  checkError(cudaMalloc((void**)&r, n*sizeof(double)));
  checkError(cudaMalloc((void**)&Ap, n*sizeof(double))); 
  checkError(cudaMalloc((void**)&z, n*sizeof(double))); 
  checkError(cudaMalloc((void**)&t, n*sizeof(double))); 

  /* allocate data array to avoid DtoH communication */
  double *data;
  checkError(cudaMalloc((void**)&data, 4*sizeof(double))); 
  double *rr_old = data, *rr_new = data+1 ;
  double *pAp = data+2;
  double *r_norm = data+3;

  /* ------------------------------------------------------------------------------------------------ */

  /* create handle for cublas e cusparse, assign stream and pointer mode device */ 
  cublasHandle_t handle;
  checkError(cublasCreate(&handle)); 
  checkError(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  checkError(cublasSetStream(handle, stream));

  cusparseHandle_t sphandle;
  checkError(cusparseCreate(&sphandle)); 
  checkError(cusparseSetPointerMode(sphandle, CUSPARSE_POINTER_MODE_DEVICE));
  checkError(cusparseSetStream(sphandle, stream));

  /* ------------------------------------------------------------------------------------------------ */

  /* create mat descriptor for A to USE FOR MATRIX-VECTOR MUTIPLICATION */
  cusparseMatDescr_t descrA;
  checkError(cusparseCreateMatDescr(&descrA));
  checkError(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC)); // <---------
  checkError(cusparseSetMatFillMode(descrA, CUSPARSE_FILL_MODE_LOWER)); // <-- need to be generalized
  checkError(cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO));
  checkError(cusparseSetMatDiagType(descrA, CUSPARSE_DIAG_TYPE_NON_UNIT));

  /* ------------------------------------------------------------------------------------------------ */

  /*
   * copy csrValA to csrValM for perform cholesky 
   * cholesky section is copied from http://docs.nvidia.com/cuda/cusparse/index.html#cusparse-lt-t-gt-csric02 
   * TODO understand better level analisys
   *
   */ 

  double *csrValM; ///< pointing to cholesky factorization 
  checkError(cudaMalloc((void**)&csrValM, nnz*sizeof(double))); 
  checkError(cudaMemcpyAsync(csrValM, csrValA, nnz*sizeof(double), cudaMemcpyDeviceToDevice, stream));

  cusparseMatDescr_t descr_M = 0;
  cusparseMatDescr_t descr_L = 0;
  csric02Info_t info_M = 0; 
  csrsv2Info_t info_L = 0;
  csrsv2Info_t info_Lt = 0;

  int pBufferSize_M; int pBufferSize_L;
  int pBufferSize_Lt; int pBufferSize;

  char *pBuffer = 0; int structural_zero; int numerical_zero;

  const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
  const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

  checkError(cusparseCreateMatDescr(&descr_M));
  checkError(cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO));
  checkError(cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL));

  checkError(cusparseCreateMatDescr(&descr_L));
  checkError(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
  checkError(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
  checkError(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
  checkError(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT)); 

  checkError(cusparseCreateCsric02Info(&info_M));
  checkError(cusparseCreateCsrsv2Info(&info_L));
  checkError(cusparseCreateCsrsv2Info(&info_Lt));

  checkError( cusparseDcsric02_bufferSize(sphandle, n, nnz, descr_M,
                                          csrValM, csrRowPtrA, csrColIndA, info_M, &pBufferSize_M) );
  
  checkError( cusparseDcsrsv2_bufferSize(sphandle, trans_L, n, nnz, descr_L,
                                         csrValM, csrRowPtrA, csrColIndA, info_L, &pBufferSize_L) );

  checkError( cusparseDcsrsv2_bufferSize(sphandle, trans_Lt, n, nnz, descr_L,
                                         csrValM, csrRowPtrA, csrColIndA, info_Lt, &pBufferSize_Lt) );

  pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));

  //std::cerr<<pBufferSize<<std::endl;
  checkError(cudaMalloc((void**)&pBuffer, pBufferSize));

  cusparseStatus_t status;

  checkError( cusparseDcsric02_analysis(sphandle, n, nnz, descr_M,
                                        csrValM, csrRowPtrA, csrColIndA, info_M, policy_M, pBuffer) );

//produce error undestand and fix
//status = cusparseXcsric02_zeroPivot(sphandle, info_M, &structural_zero);
//checkError(status);
//if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("A(%d,%d) is missing\n", structural_zero, structural_zero); } 

  checkError( cusparseDcsrsv2_analysis(sphandle, trans_L, n, nnz, descr_L,
                                       csrValM, csrRowPtrA, csrColIndA, info_L, policy_L, pBuffer) ); 
  checkError( cusparseDcsrsv2_analysis(sphandle, trans_Lt, n, nnz, descr_L,
                                       csrValM, csrRowPtrA, csrColIndA, info_Lt, policy_Lt, pBuffer) );

  checkError( cusparseDcsric02(sphandle, n, nnz, descr_M,
                               csrValM, csrRowPtrA, csrColIndA, info_M, policy_M, pBuffer) );
 
//same previous
//status = cusparseXcsric02_zeroPivot(sphandle, info_M, &numerical_zero);
//checkError(status);
//if (CUSPARSE_STATUS_ZERO_PIVOT == status){ printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero); }


  /* ------------------------------------------------------------------------------------------------ */

  /* try to provide a better x0 from icomplete cholesky */

  checkError( cusparseDcsrsv2_solve(sphandle, trans_L,
                                    n, nnz, one, descr_L, csrValM, csrRowPtrA, csrColIndA,
                                    info_L, b, t, policy_L, pBuffer) );

  checkError( cusparseDcsrsv2_solve(sphandle, trans_Lt,
                                    n, nnz, one, descr_L, csrValM, csrRowPtrA, csrColIndA,
                                    info_Lt, t, x, policy_Lt, pBuffer) );


  /* ------------------------------------------------------------------------------------------------ */

  /* inizialize r: r=b-Ax */
  // r = -Ax

  checkError( cusparseDcsrmv(sphandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             n, n, nnz, minus_one, descrA, csrValA, csrRowPtrA, csrColIndA, x, zero, r) );
  // r = b + r
  checkError(cublasDaxpy(handle, n, one, b, 1, r, 1));

  // z = M * r

  checkError( cusparseDcsrsv2_solve(sphandle, trans_L,
                                    n, nnz, one, descr_L, csrValM, csrRowPtrA, csrColIndA,
                                    info_L, r, t, policy_L, pBuffer) );

  checkError( cusparseDcsrsv2_solve(sphandle, trans_Lt,
                                    n, nnz, one, descr_L, csrValM, csrRowPtrA, csrColIndA,
                                    info_Lt, t, z, policy_Lt, pBuffer) );

  /* p = z */
  checkError(cublasDcopy(handle, n, z, 1, p, 1));

  /* compute r*z */
  checkError(cublasDdot(handle, n, r, 1, z, 1, rr_old)); 

  /* compute r_norm */
  checkError(cublasDnrm2(handle, n, r, 1, r_norm));

  /* ------------------------------------------------------------------------------------------------ */

  double ctol;
  checkError(cudaMemcpyAsync(&ctol, r_norm, sizeof(double), cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  
  unsigned int k = 0;
//std::cerr<<k<<"\t"<<ctol<<std::endl;

  while(ctol>tol && k<kmax){

    // compute Ap
    checkError( cusparseDcsrmv(sphandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               n, n, nnz, one, descrA, csrValA, csrRowPtrA, csrColIndA, p, zero, Ap) );
    cublasDdot(handle, n, p, 1, Ap, 1, pAp);

    // alpha kernel 
    alpha_kernel<<<nob, tpb,0,stream>>>(n, x, r, p, Ap, rr_old, pAp);

    // z = M * r

    checkError( cusparseDcsrsv2_solve(sphandle, trans_L,
                                      n, nnz, one, descr_L, csrValM, csrRowPtrA, csrColIndA,
                                      info_L, r, t, policy_L, pBuffer) );

    checkError( cusparseDcsrsv2_solve(sphandle, trans_Lt,
                                      n, nnz, one, descr_L, csrValM, csrRowPtrA, csrColIndA,
                                      info_Lt, t, z, policy_Lt, pBuffer) );

    // z=r;
    checkError(cublasDdot(handle, n, r, 1, z, 1, rr_new));
 
    checkError(cublasDnrm2(handle, n, r, 1, r_norm));

    // beta kernel 
    beta_kernel<<<nob, tpb,0,stream>>>(n, p, z, rr_new, rr_old);

    ++k;

    cudaMemcpyAsync(&ctol, r_norm, sizeof(double), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  //std::cerr<<k<<"\t"<<ctol<<std::endl;

  }   


  cusparseDestroyMatDescr(descrA);
  cublasDestroy(handle);
  cusparseDestroy(sphandle);
  cudaFree(p);
  cudaFree(r);
  cudaFree(Ap);
  cudaFree(z);
  cudaFree(t);
  cudaFree(csrValM);

  // add destructor

  return k;
}

#endif
 
