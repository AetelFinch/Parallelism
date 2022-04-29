all: compile

compile:
	pgcc cuBLAS_poisson.c -acc -Minfo=accel -ta=nvidia -cudalib=cublas
