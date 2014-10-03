#ifndef __CUDATIMER_HPP__
#define __CUDATIMER_HPP__

//http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g14c387cc57ce2e328f6669854e6020a5


/**
 simple timer measure the time from the creation to the distruction
 of the object, waiting the ending of the gpu computation
*/
class CUDATIMER{
  private:
   cudaEvent_t start, stop;
   std::string label;

  public:
   CUDATIMER(std::string l):label(l) {

     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start,0);

   }


   ~CUDATIMER(){
     cudaEventRecord(stop,0);
     cudaEventSynchronize(stop);
     float et;
     cudaEventElapsedTime(&et,start,stop);
     cudaEventDestroy(start); 
     cudaEventDestroy(stop);
     std::cerr<<label<<": "<<et<<"ms"<<std::endl;
     /* time in milliseconds with a resolution of around 0.5 microseconds */
   }

};

#endif

