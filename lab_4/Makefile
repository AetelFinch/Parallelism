all: compile profile

compile:
	nvcc cuda_poisson.cu -arch sm_70

profile:
	nsys profile -o prof -f true -t cuda ./a.out 0.000001 512 150
	nsys stats prof.nsys-rep
