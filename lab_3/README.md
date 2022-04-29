# Heat Equation with openACC + optimized libraries

## task

Implement the solution of the heat equation (five-point pattern) in a two-dimensional domain

Boundary conditions – linear interpolation between the corners of the region. The value in the corners is 10, 20, 30, 20

Parameters (accuracy, grid size, number of iterations) must be set via command line parameters

The output of the program is the number of iterations and the achieved error value

Implement reduction operation (calculation of the maximum error value) on the GPU through function calls from the cuBLAS library

## compilation

```
make compile
```