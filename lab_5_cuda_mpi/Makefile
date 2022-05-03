compile:
	nvcc -I/usr/local/openmpi/include -L/usr/local/openmpi/lib -lmpi -lnccl -arch sm_70 mpi_nccl_poisson.cu

run:
	UCX_WARN_UNUSED_ENV_VARS=n mpirun -mca pml ucx -x UCX_TLS=cuda,sm,posix -np 1 ./a.out 0.000001 128 1000000

profile:
	nsys profile -o prof -f true -t cuda,mpi mpirun -np 4 ./a.out 0.000001 1024 1000