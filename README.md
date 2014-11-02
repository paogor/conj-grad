Just another conjugate gradient solver written in CUDA,
using cublas and cusparse.

Function is written using streams. So it is possible
run multiple solver concurrently on the same device.

_CUDA 6.0 required._


