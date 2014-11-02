#ifndef __CUDATIMER_HPP__
#define __CUDATIMER_HPP__

//16.10.14

//http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g14c387cc57ce2e328f6669854e6020a5


/**
 simple timer measure the time from the creation to the distruction
 of the object, waiting the ending of the gpu computation
*/
class CUDATIMER
{
  private:
   cudaEvent_t _start, _stop;
   std::string label;
   float * elapsed_time_pointer;
   float elapsed_time_ms;  /**< time in milliseconds with resolution
                                of around 0.5 microseconds */

   void start()
   {
     cudaEventCreate(&_start);
     cudaEventCreate(&_stop);
     cudaEventRecord(_start,0);
   }

   void stop()
   {
     cudaEventRecord(_stop,0);
     cudaEventSynchronize(_stop);

     cudaEventElapsedTime(&elapsed_time_ms,_start,_stop);
     cudaEventDestroy(_start); 
     cudaEventDestroy(_stop);
   }


  public:

   CUDATIMER(std::string l):label(l), elapsed_time_pointer(NULL)
   {
     start();
   }

   CUDATIMER(float *elaps_pntr): elapsed_time_pointer(elaps_pntr)
   {
     start();
   }

   ~CUDATIMER()
   {
     stop();
     if(elapsed_time_pointer == NULL)
       std::cout<<label<<": "<<elapsed_time_ms<<"ms"<<std::endl;
       else *elapsed_time_pointer = elapsed_time_ms;
   }

};

#endif

