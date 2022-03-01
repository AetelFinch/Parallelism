all: res

res: cuBLAS_poisson.c
	pgcc cuBLAS_poisson.c -acc -Minfo=accel -ta=nvidia -cudalib=cublas
