Just another conjugate gradient solver written in CUDA,
using cublas and cusparse.

Function are written using streams. So it is possible
run multiple solve concurrently on the same device.

CUDA 6.0 required.


