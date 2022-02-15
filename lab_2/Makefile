all: res

res: poisson_equation.c
	pgcc poisson_equation.c -acc -Minfo=accel -ta=tesla:managed
