ARCH=sm_20

test_simple: test_simple.cu *.hpp 
	nvcc test_simple.cu -o=test_simple -arch=$(ARCH) -lcublas -lcusparse
	
test_multiples_streams: test_multiples_streams.cu *.hpp
	nvcc test_multiples_streams.cu -o=test_multiples_streams -arch=$(ARCH) -lcublas -lcusparse -Xcompiler -fopenmp
