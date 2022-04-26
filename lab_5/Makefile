compile:
	nvcc -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lmpi -arch sm_70 mpi_poisson.cu

run:
	UCX_WARN_UNUSED_ENV_VARS=n mpirun -np 4 ./a.out 0.000001 1024 1000000