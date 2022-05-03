#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define BLOCK_DIM 16
#define BLOCK_VEC_DIM 256

#define CUDACHKERR(err) if (err != cudaSuccess) { \
    fprintf(stderr, \
            "Failed to copy vector B from host to device (error code %s)!\n", \
            cudaGetErrorString(err)); \
    exit(EXIT_FAILURE); \
    }

void print_help()
{
    printf("usage:\n");
    printf("{min_error} {matrix_size} {iter_max}\n");
}

double* getSetMatrix(double* dst, int size, cudaStream_t stream)
{
    cudaError_t err;

    double *matrix;
    err = cudaMalloc(&matrix, size * size * sizeof(double));
    CUDACHKERR(err);

    err = cudaMemcpyAsync(matrix, dst, size * size * sizeof(double), cudaMemcpyHostToDevice, stream);
    CUDACHKERR(err);

    return matrix;
}

void interpolationMatrixSides(double* matrix, int matrix_size)
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

__global__ void vecNeg(const double *newA, const double *A, double* ans, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numElements)
    {
        ans[idx] =  newA[idx] - A[idx];
    }
}

__global__ void evalEquation(double *newA, const double *A, int numElements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if ((0 < idx && idx < numElements - 1) && (0 < idy && idy < numElements - 1))
    {
        newA[idy * numElements + idx] = 0.25 * (__ldg(&A[(idy - 1) * numElements + idx]) + __ldg(&A[(idy + 1) * numElements + idx]) +
                                                __ldg(&A[idy * numElements + (idx - 1)]) + __ldg(&A[idy * numElements + (idx + 1)]));
    }
}

void printCudaMatrix(double* dst, int size)
{
    double *a = (double*)calloc(sizeof(double), size * size);

    cudaMemcpy(a, dst, size * size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            printf("%lf ", a[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");

    free(a);
}

void checkCudaInfo()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    printf("major = %d \n", prop.major);
    printf("warp Size = %d \n", prop.warpSize);
    printf("max Threads Per Block = %d \n", prop.maxThreadsPerBlock);
    printf("max Threads Per MultiProcessor = %d \n", prop.maxThreadsPerMultiProcessor);
    printf("multiProcessor Count = %d \n", prop.multiProcessorCount);
    printf("shared Memory Per Block (bytes) = %lu \n", prop.sharedMemPerBlock);
    printf("max Grid Size by X = %d \n", prop.maxGridSize[0]);
    printf("max Grid Size by Y = %d \n", prop.maxGridSize[1]);

    printf("\n");
}

int main(int argc, char *argv[])
{
    if (argc == 1)
    {
        checkCudaInfo();
        print_help();
        exit(0);
    }

    double min_error = atof(argv[1]);
    int matrix_size = atoi(argv[2]);
    int iter_max = atoi(argv[3]);

    cudaError_t err;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    double *tmp = (double*)calloc(sizeof(double), matrix_size * matrix_size);

    tmp[0] = 10.0;
    tmp[matrix_size - 1] = 20.0;
    tmp[(matrix_size - 1) * matrix_size] = 20.0;
    tmp[(matrix_size - 1) * matrix_size + matrix_size - 1] = 30.0;

    interpolationMatrixSides(tmp, matrix_size);

    double *A_d = getSetMatrix(tmp, matrix_size, stream);
    double *newA_d = getSetMatrix(tmp, matrix_size, stream);
    free(tmp);

    int iter = 0;
    double error = 10;

    dim3 BS = dim3(BLOCK_DIM, BLOCK_DIM);

    dim3 GS = dim3(ceil(matrix_size / (double)BS.x), ceil(matrix_size / (double)BS.y));

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    double *tmp_d, *max_d;
    cudaMalloc(&tmp_d, sizeof(double) * matrix_size * matrix_size);
    cudaMalloc(&max_d, sizeof(double));

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, matrix_size * matrix_size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    int BS_neg = BLOCK_VEC_DIM;
    int GS_neg = ceil(matrix_size * matrix_size / (double)BS_neg);

    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStreamSynchronize(stream);

    while (error > min_error && iter < iter_max)
    {
        if (!graphCreated)
        {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            for (int i = 0; i < 100; ++i)
            {
                evalEquation<<<GS, BS, 0, stream>>>(newA_d, A_d, matrix_size);
                tmp = A_d;
                A_d = newA_d;
                newA_d = tmp;
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }
        err = cudaGraphLaunch(instance, stream);
        CUDACHKERR(err);
        err = cudaStreamSynchronize(stream);
        CUDACHKERR(err);

        iter += 100;

        if (iter % 100 == 0)
        {
            printf("iter = %d error = %e\n", iter, error);
            error = 0;

            tmp = A_d;
            A_d = newA_d;
            newA_d = tmp;

            vecNeg<<<GS_neg, BS_neg, 0, stream>>>(newA_d, A_d, tmp_d, matrix_size * matrix_size);

            err = cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, tmp_d, max_d, matrix_size * matrix_size, stream);
            CUDACHKERR(err);

            err = cudaMemcpyAsync(&error, max_d, sizeof(double), cudaMemcpyDeviceToHost, stream);
            CUDACHKERR(err);

            tmp = A_d;
            A_d = newA_d;
            newA_d = tmp;
        }
    }

    cudaFree(A_d);
    cudaFree(newA_d);
    cudaFree(tmp_d);
    cudaFree(max_d);
    cudaFree(d_temp_storage);

    return 0;
}
