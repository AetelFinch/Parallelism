compile:
	nvcc -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lmpi -arch sm_70 mpi_poisson.cu

run:
	UCX_WARN_UNUSED_ENV_VARS=n mpirun -mca pml ucx -x UCX_TLS=cuda,sm,posix -np 4 ./a.out 0.000001 128 1000000