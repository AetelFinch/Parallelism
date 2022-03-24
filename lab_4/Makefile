all: res

res: cuda_poisson.cu
	nvcc cuda_poisson.cu -arch sm_70

prof: ./a.out
	nsys profile -o prof -f true -t cuda ./a.out 0.000001 128 150
	nsys stats prof.nsys-rep