#ifndef __CUDA_ERRORS_HPP__
#define __CUDA_ERRORS_HPP__

#include<iostream> /** for cerr */
#include<string> 
#include<cstdio>

// for debugging
__global__ void print_array(const double *a, int n){

 for(int i =0; i<n;++i)
  printf("%f\t",a[i]);
 printf("\n");

}

__global__ void print_array(const int *a, int n){

 for(int i =0; i<n;++i)
  printf("%d\t",a[i]);
 printf("\n");

}


namespace CUDA_ERRORS
{
  std::string _cudaGetErrorEnum(cudaError_t error)
  {
    switch (error)
    {
      case cudaSuccess:
        return "cudaSuccess";
      case cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration";
      case cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation";
      case cudaErrorInitializationError:
        return "cudaErrorInitializationError";
      case cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure";
      case cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure";
      case cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout";
      case cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources";
      case cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction";
      case cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration";
      case cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice";
      case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";
      case cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue";
      case cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol";
      case cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed";
      case cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed";
      case cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer";
      case cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer";
      case cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture";
      case cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding";
      case cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor";
      case cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection";
      case cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant";
      case cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed";
      case cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound";
      case cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError";
      case cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting";
      case cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting";
      case cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution";
      case cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading";
      case cudaErrorUnknown:
        return "cudaErrorUnknown";
      case cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented";
      case cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge";
      case cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle";
      case cudaErrorNotReady:
        return "cudaErrorNotReady";
      case cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver";
      case cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess";
      case cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface";
      case cudaErrorNoDevice:
        return "cudaErrorNoDevice";
      case cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable";
      case cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound";
      case cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed";
      case cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit";
      case cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName";
      case cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName";
      case cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName";
      case cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable";
      case cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage";
      case cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice";
      case cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext";
      case cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled";
      case cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled";
      case cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse";
      case cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled";
      case cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized";
      case cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted";
      case cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped";
#if __CUDA_API_VERSION >= 0x4000
      case cudaErrorAssert:
        return "cudaErrorAssert";
      case cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers";
      case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";
      case cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered";
#endif
      case cudaErrorStartupFailure:
        return "cudaErrorStartupFailure";
      case cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
  }

  std::string _cudaGetErrorEnum(cublasStatus_t error)
  {
    switch (error)
    {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
       return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
  }

  std::string _cudaGetErrorEnum(cusparseStatus_t error)
  {
    switch (error)
    {
      case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";
      case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";
      case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";
      case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";
      case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";
      case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";
      case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";
      case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";
      case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    }

    return "<unknown>";
  }

  template< typename T >
  void check( T result, 
              char const *const func,
              const char *const file,
              int const line )
  {
    if (result)
    {
       std::cerr<<"CUDA ERROR "<<file<<"@"<<line<<" - ";
       std::cerr<<_cudaGetErrorEnum(result)<<"\n\t"<<func<<std::endl;

       cudaDeviceReset();
       // Make sure we call CUDA Device Reset before exiting
       exit(EXIT_FAILURE);
    }
  }

}

#define checkError(val) CUDA_ERRORS::check( (val), #val, __FILE__, __LINE__ )

#endif 
