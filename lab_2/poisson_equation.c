#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void print_help()
{
	printf("usage:\n");
	printf("{min_error} {matrix_size} {iter_max}\n");
}

void print_matrix(double* matrix, int matrix_size)
{
	for (int i = 0; i < matrix_size; ++i)
	{
		for (int j = 0; j < matrix_size; ++j)
		{
			printf("%lf ", matrix[i * matrix_size + j]);
		}
		printf("\n");
	}
	printf("\n");
}

void save_matrix(double** matrix, int matrix_size, char* filename)
{
	FILE *file = fopen(filename, "w");

	for (int i = 0; i < matrix_size; ++i)
	{
		for (int j = 0; j < matrix_size; ++j)
		{
			fprintf(file, "%lf ", matrix[i][j]);
		}
		fprintf(file, "\n");
	}

	fclose(file);
}

double* get_matrix(int matrix_size)
{
	double *matrix = (double*)malloc(sizeof(double) * matrix_size * matrix_size);
	return matrix;
}

void interpolation_matrix_sides(double* matrix, int matrix_size)
{
#pragma acc kernels
{ 
	// left side
	for (int i = 1; i < matrix_size - 1; ++i)
	{
		matrix[i * matrix_size] = matrix[0] * (matrix_size - 1 - i) / (matrix_size - 1) +
					   			  matrix[matrix_size * (matrix_size - 1)] * i / (matrix_size - 1);
	}

	// top side
	for (int i = 1; i < matrix_size - 1; ++i)
	{
		matrix[i] = matrix[0] * (matrix_size - 1 - i) / (matrix_size - 1) +
					   matrix[matrix_size - 1] * i / (matrix_size - 1);
	}

	// right side
	for (int i = 1; i < matrix_size - 1; ++i)
	{
		matrix[i * matrix_size + matrix_size - 1] = matrix[matrix_size - 1] * (matrix_size - 1 - i) / (matrix_size - 1) +
					   				 matrix[(matrix_size - 1) * matrix_size + matrix_size - 1] * i / (matrix_size - 1);
	}

	// bottom side
	for (int i = 1; i < matrix_size - 1; ++i)
	{
		matrix[(matrix_size - 1) * matrix_size + i] = matrix[(matrix_size - 1) * matrix_size] * (matrix_size - 1 - i) / (matrix_size - 1) +
					                 				  matrix[(matrix_size - 1) * matrix_size + matrix_size - 1] * i / (matrix_size - 1);
	}
}
}

int main(int argc, char *argv[])
{
	if (argc == 1)
	{
		print_help();
		exit(0);
	}

	double min_error = atof(argv[1]);
	int matrix_size = atoi(argv[2]);
	int iter_max = atoi(argv[3]);

	// initialize matrix with start conditions
	double *matrix = get_matrix(matrix_size);

	matrix[0] = 10.0;
	matrix[matrix_size - 1] = 20.0;
	matrix[(matrix_size - 1) * matrix_size] = 20.0;
	matrix[(matrix_size - 1) * matrix_size + matrix_size - 1] = 30.0;

	// create new buffer matrix
	double *new_matrix = get_matrix(matrix_size);

	interpolation_matrix_sides(matrix, matrix_size);

	int iter = 0;
	double error = 100;

	while (error > min_error && iter < iter_max)
	{
		++iter;
		error = 0;

#pragma acc kernels
{

		for (int row_i = 1; row_i < matrix_size - 1; ++row_i)
		{
			for (int col_i = 1; col_i < matrix_size - 1; ++col_i)
			{
				new_matrix[row_i * matrix_size + col_i] = 0.25 * (matrix[(row_i - 1) * matrix_size + col_i] + matrix[(row_i - 1) * matrix_size + col_i] +
								   								  matrix[row_i * matrix_size + (col_i - 1)] + matrix[row_i * matrix_size + (col_i + 1)]);

				error = fmax(error, new_matrix[row_i * matrix_size + col_i] - matrix[row_i * matrix_size + col_i]);
			}
		}

		for (int row_i = 1; row_i < matrix_size - 1; ++row_i)
		{
			for (int col_i = 1; col_i < matrix_size - 1; ++col_i)
			{
				matrix[row_i * matrix_size + col_i] = new_matrix[row_i * matrix_size + col_i];
			}
		}
}
	}
	printf("iter = %d\n", iter);
	printf("error = %e\n", error);
	// save_matrix(matrix, matrix_size, "matrix.txt");

	return 0;
}

