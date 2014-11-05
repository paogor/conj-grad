conj-grad
---------

Just another _conjugate gradient_ solver function written in CUDA,
using cuBLAS and cuSPARSE.  
CUDA 6.0 or higher required.

### function

`int cg_sparse(cudaStream_t stream,
               const double *csrValA, const int *csrRowPtrA, const int *csrColIndA,
               double *b, double *x, const int n, const int nnz, double tol=1e-9);`

- `stream` witch the solver is launched. So it is possible
run multiple solver concurrently on the same device;  
- `csrValA`, `csrRowPtrA` and `csrColIndA`: arrays representing the matrix
stored in Compressed Storage Row sparse format;  
- `b`: rhs array;  
- `x`: solution array;  
- `n`: size of arrays;  
- `nnz`: non-zero matrix elements; 
- `tol`: tolerance to stop iterations.  

The function return the number of iterations.             
               
### test usage

`$ make test_simple`  
`$ ./test_simple`  

`$ make test_multiples_streams`  
`$ ./test_multiples_streams [# streams launched concurrently]`
