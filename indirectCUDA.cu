//
// Author     :  matto@xilinx 14JAN2018, alai@xilinx 25JULY2018
// Filename   :  indirectTest_onlyGPU.cu
// Description:  Cuda random access benchmark example based on indirect.c by gswart/skchavan@oracle
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <utime.h>
#include <sys/time.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>

//#define DEBUG
#define CPU_BENCH
#define NOCUDA

#define MEM_LOGN 28
//#define GATHER2

#define FULLMEM
//#define VERIF

enum {
  rows = 1U << 22,
  array = 1U << 22,
  groups = 1U << 10,
  segment_bits = 12,
  segments = array / (1U << segment_bits)
};

struct Row {
  unsigned int measure;
  unsigned int group;
};

struct Row16 {
  // each Row stucture is 8 bytes
  //struct Row rows_arr[16]; // 128 bytes
  //struct Row rows_arr[4]; // 32 bytes
  
  // [input size/size of Row]
  struct Row rows_arr[32/8];
};

struct String {
  char str[128];
};
	
#ifdef NOCUDA
// ikimasu
//__device__ struct Row d_A[array];
//__device__ unsigned int d_in[rows];
//__device__ struct Row d_out[rows];
//__device__ unsigned long long d_agg1[groups];
//__device__ unsigned long long d_agg2[groups];
//__device__ struct Row d_out2[rows];
//__device__ struct Row * d_B[segments];

__device__ struct Row16 dd_A[array];
__device__ unsigned int dd_in[rows];
__device__ struct Row16 dd_out[rows];
__device__ struct Row16 dd_out2[rows];

// initialize the GPU arrays
__global__ void d_init()
{
    printf("Initializing data structures.\n");
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);
    curandState state;
    curand_init((unsigned long long)clock() + tId, 0, 0, &state);
    //printf("Size of word: %lu bytes\n", (unsigned long)sizeof(dd_A[0].str));
    //printf("Size of word container: %lu bytes\n", (unsigned long)sizeof(dd_A[0]));
    unsigned long input_size = (unsigned long)sizeof(dd_A[0]);

    // Random fill indirection array A
    unsigned int i;
    unsigned int j;
    printf("Randomly filling array A.\n");
    for (i = 0; i < array; i++) {
      for (j = 0; j < (input_size/8); j++) {
        dd_A[i].rows_arr[j].measure = curand_uniform(&state) * array;
        dd_A[i].rows_arr[j].group = curand_uniform(&state) * groups;
        //printf("dd_A[%d][%d] - %d\n",i,j,dd_A[i].rows_arr[j].measure);
      }
    }
    printf("Size of row container: %lu bytes\n", (unsigned long)sizeof(dd_A[0]));

    // Random fill input
    printf("Random filling input array.\n");
    for (i = 0; i < rows; i++) {
        dd_in[i] = curand_uniform(&state) * array;
        //printf("dd_in[%d] - %d\n",i,dd_in[i]);
    }
    printf("Successfully initialized input array.\n");

    // generate random array for benching writes
    for (i = 0; i < rows; i++) {
      dd_out[i] = dd_out2[dd_in[i]];
    }
}

// bench gathers
__global__ void d_bench()
{
  unsigned i;
  for (i = 0; i < rows; i++) {
    dd_out[i] = dd_A[dd_in[i]];
  }
  
}

// bench linear reads
__global__ void d_bench_read_linear()
{
  unsigned i;
  for (i = 0; i < rows; i++) {
    dd_A[i];
  }
}

// bench random reads
__global__ void d_bench_read_random()
{
  unsigned i;
  for (i = 0; i < rows; i++) {
    dd_A[dd_in[i]];
  }
}

// bench linear writes
__global__ void d_bench_write_linear()
{
  unsigned i;
  for (i = 0; i < rows; i++) {
    dd_out[i] = dd_out2[i];
  }
}

// bench random writes
__global__ void d_bench_write_random()
{
  unsigned i;
  for (i = 0; i < rows; i++) {
    dd_out[i] = dd_out2[dd_in[i]];
  }
}

#endif // !1

#ifdef VERIF
static __global__ void
d_check(size_t n, benchtype *t)
{
	for (i = 0; i < groups; i++) {
		if (d_agg1[i] != d_agg2[i]) printf("Agg doesn't match: %d\n", i);
	}
}
#endif // VERIF

int main(int argc, char** argv) {
#ifdef NOCUDA
  int ndev;
  cudaGetDeviceCount(&ndev);
  int dev = 0;
  //unsigned num_sm = 1; // 1, 2, 4, 8 // # of SMs

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, dev);
  cudaSetDevice(dev);

  printf("Using GPU %d of %d GPUs.\n", dev, ndev);
  printf("Warp size = %d.\n", prop.warpSize);
  printf("Multi-processor count = %d.\n", prop.multiProcessorCount);
  printf("Max threads per multi-processor = %d.\n", prop.maxThreadsPerMultiProcessor);
  printf("Grid Size = %d.\n", prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.warpSize));
  printf("Thread Size = %d.\n", prop.warpSize);

  dim3 grid(prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / prop.warpSize));
  dim3 thread(prop.warpSize);

  unsigned long input_size = (unsigned long)sizeof(dd_A[0]);
  printf("Size of word container: %lu bytes\n", input_size);
  //printf("Number of SMs: %d\n", num_sm);

  printf("Initializing arrays on GPU with %d elements.\n", array);
  // << <# blocks per grid, # threads per block> >>
  // max = << <65536,1024> >>
  d_init << <8192, 2048>> >();
  //d_bench_write_initialize << <8192, 2048>> >();


  // single threaded
  cudaEvent_t read_begin, read_end, write_begin, write_end;
  cudaEventCreate(&read_begin);
  cudaEventCreate(&read_end);
  cudaEventCreate(&write_begin);
  cudaEventCreate(&write_end);

  float ms_read_linear, ms_write_linear, ms_read_random, ms_write_random;

  // linear read/write
  printf("Benching linear reads.\n");
  cudaEventRecord(read_begin);
  cudaEventSynchronize(read_begin);
  d_bench_read_linear << <1, 1>> >();
  cudaEventRecord(read_end);
  cudaEventSynchronize(read_end);

  printf("Benching linear writes.\n");
  cudaEventRecord(write_begin);
  cudaEventSynchronize(write_begin);
  d_bench_write_linear << <1, 1>> >();
  cudaEventRecord(write_end);
  cudaEventSynchronize(write_end);

  cudaEventElapsedTime(&ms_read_linear, read_begin, read_end);
  cudaEventElapsedTime(&ms_write_linear, write_begin, write_end);
  printf("%lu-byte linear read average = %.6f ms.\n", input_size, (ms_read_linear)/rows);
  printf("%lu-byte linear write average = %.6f ms.\n", input_size, (ms_write_linear)/rows);

  // random read/write
  printf("Benching random reads.\n");
  cudaEventRecord(read_begin);
  cudaEventSynchronize(read_begin);
  d_bench_read_random << <1, 1>> >();
  cudaEventRecord(read_end);
  cudaEventSynchronize(read_end);

  printf("Benching random writes.\n");
  cudaEventRecord(write_begin);
  cudaEventSynchronize(write_begin);
  d_bench_write_random << <1, 1>> >();
  cudaEventRecord(write_end);
  cudaEventSynchronize(write_end);

  cudaEventElapsedTime(&ms_read_random, read_begin, read_end);
  cudaEventElapsedTime(&ms_write_random, write_begin, write_end);
  printf("%lu-byte random read average = %.6f ms.\n", input_size, (ms_read_random)/rows);
  printf("%lu-byte random write average = %.6f ms.\n", input_size, (ms_write_random)/rows);

  cudaEventDestroy(write_end);
  cudaEventDestroy(write_begin);
  cudaEventDestroy(read_end);
  cudaEventDestroy(read_begin);

  printf("Elapsed time = %.6f seconds.\n", (ms_read_linear + ms_write_linear + ms_read_random + ms_write_random)/1000);

  //double time = ms * 1.0e-3;
  //printf("GPU elapsed time = %.6f seconds.\n", time);
  

#endif // !1

#ifdef VERIF
  //d_check << <grid, thread >> >(n, d_t);
  //cpu_bench();
#endif // VERIF

  /**
  printf("Copying host arrays from device.\n");
  checkCudaErrors(cudaMemcpyFromSymbol(h_A, d_A, sizeof(d_A)));
  //checkCudaErrors(cudaMemcpyFromSymbol(h_B, d_B, sizeof(d_B)));
  checkCudaErrors(cudaMemcpyFromSymbol(h_in, d_in, sizeof(d_in)));
  //checkCudaErrors(cudaMemcpyFromSymbol(h_out, d_out, sizeof(d_out)));
  //checkCudaErrors(cudaMemcpyFromSymbol(h_out2, d_out2, sizeof(d_out2)));
  //checkCudaErrors(cudaMemcpyFromSymbol(h_agg1, d_agg1, sizeof(d_agg1)));
  //checkCudaErrors(cudaMemcpyFromSymbol(h_agg2, d_agg2, sizeof(d_agg2)));
  printf("Successfully copied GPU arrays.\n");**/

#ifdef NOCUDA

  //cudaFree(d_A);
  //cudaFree(d_in);
  //cudaFree(d_out);
  //cudaFree(d_out2);
  //cudaFree(d_agg1);
  //cudaFree(d_agg2);

  cudaFree(dd_A);
  cudaFree(dd_in);
  cudaFree(dd_out);
  cudaFree(dd_out2);

#endif // !1
  //unsigned i;
/**
#ifdef CPU_BENCH
  printf("Beginning CPU benchmark.\n");
  struct timeval t0, t1;
  gettimeofday(&t0, 0);
  // Gather rows
  for (i = 0; i < rows; i++) {
          h_out[i] = h_A[h_in[i]];
  }
  // Indirect Gather rows
  for (i = 0; i < rows; i++) {
          h_out[i] = h_A[h_A[h_in[i]].measure];
  }

  // Fused gather group
  for (i = 0; i < rows; i++) {
          h_agg2[h_A[h_in[i]].group] += h_A[h_in[i]].measure;
#ifdef DEBUG
          printf("CPU:  h_agg2[h_A[h_in[i]].group]  = %d\n", h_agg2[h_A[h_in[i]].group]);
#endif // DEBUG
  }
  gettimeofday(&t1, 0);
  printf("CPU bench successful.\n");
  long elapsed = ((t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec);
  printf("CPU elapsed time = %lu microseconds.\n", elapsed);

#endif // CPU_BENCH
**/
  return 0;
}
