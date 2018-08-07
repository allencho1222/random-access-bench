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

#ifdef FULLMEM
enum {
	rows = 1U << 26,
  array = 1U << 26,
	groups = 1U << 10,
	segment_bits = 12,
	segments = array / (1U << segment_bits)
};
#else // FULLMEM
enum {
	rows = 1U << 6,
	array = 1U << 6,
	groups = 1U << 4,
	segment_bits = 6,
	segments = array / (1U << segment_bits)
};
#endif
struct Row {
	unsigned int measure;
	unsigned int group;
};


	
#ifdef NOCUDA
// ikimasu
__device__ struct Row d_A[array];
__device__ unsigned int d_in[rows];
__device__ struct Row d_out[rows];
__device__ unsigned long long d_agg1[groups];
__device__ unsigned long long d_agg2[groups];
__device__ struct Row d_out2[rows];
//__device__ struct Row * d_B[segments];

// initialize the GPU arrays
__global__ void d_init()
{
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);
    curandState state;
    curand_init((unsigned long long)clock() + tId, 0, 0, &state);

    printf("Initializing data structures.\n");

    // Random fill indirection array A
    unsigned int i;
    printf("Randomly filling array A.\n");
    for (i = 0; i < array; i++) {
        d_A[i].measure = curand_uniform(&state) * array;
        d_A[i].group = curand_uniform(&state) * groups;

        //printf("d_A[%d] - %d\n",i,d_A[i].measure);
    }

    // Fill segmented array B
    /**for (i = 1; i <= segments; i++) {
        d_B[i] = &(d_A[i * (1U << segment_bits)]);
    }**/

    // Random fill input
    printf("Random filling input array.\n");
    for (i = 0; i < rows; i++) {
        d_in[i] = curand_uniform(&state) * array;

        //printf("d_in[%d] - %d\n",i,d_in[i]);
    }
    printf("Successfully initialized input array.\n");
}

__global__ void d_bench()
{
    // ikimasu
    //printf("Initializing benchmarks.\n");
	unsigned i;

    // Gather rows
    printf("Beginning gather\n");
	for (i = 0; i < rows; i++) {
		d_out[i] = d_A[d_in[i]];
	}

    /**
	// Indirect Gather rows
	for (i = 0; i < rows; i++) {
		d_out[i] = d_A[d_A[d_in[i]].measure]; 
	}

	// Fused gather group
	for (i = 0; i < rows; i++) {
		d_agg2[d_A[d_in[i]].group] += d_A[d_in[i]].measure;
#ifdef DEBUG
		printf("GPU: d_agg2[d_A[d_in[i]].group]  = %d\n", d_agg2[d_A[d_in[i]].group] );
#endif // DEBUG
	}**/

    /**
#ifdef GATHER2
	// Segmented gather
	for (i = 0; i < rows; i++) {
		int segment_number = (d_in[i] >> segment_bits);
		int segment_offset = (d_in[i] & ((1U << segment_bits) - 1));
#ifdef DEBUG
		printf("d_in[i] = %d\n", d_in[i]);
		printf("segment_number = %d\n", segment_number);
		printf("segment_offset = %d\n", segment_offset);
		printf("d_B[0] = %d\n", d_B[0]);
		printf("d_B[segment_number][segment_offset] = %d\n", d_B[segment_number][segment_offset]);
		printf("d_out2[i] = %d\n", d_out2[i]);
#endif // DEBUG

		d_out2[i] = d_B[segment_number][segment_offset];
	}
#endif // GATHER2
**/

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

/**
  struct Row h_A[array];

  unsigned int h_in[rows];
  struct Row h_out[rows];
  unsigned long long h_agg1[groups];
  unsigned long long h_agg2[groups];

  struct Row h_out2[rows];
  //struct Row * h_B[segments];
**/
  //static unsigned long diff(const struct timeval * newT, const struct timeval * oldT) {
  //  return (newT->tv_sec - oldT->tv_sec)*1000000 + (newT->tv_usec - oldT->tv_usec);
  //}
int main() {

#ifdef NOCUDA
  int ndev;
  cudaGetDeviceCount(&ndev);
  int dev = 0;

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

  printf("Initializing GPU.\n");
  d_init << <8192, 2048>> >();


  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);
  cudaEventSynchronize(begin);

  //d_bench << <grid, thread >> >();
  printf("Beginning GPU benchmark.\n");
  d_bench << <8192, 2048 >> >();

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms;
  cudaEventElapsedTime(&ms, begin, end);
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  //double time = ms * 1.0e-3;
  //printf("GPU elapsed time = %.6f seconds.\n", time);
  printf("GPU elapsed time = %.6f ms.\n", ms);

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

  cudaFree(d_A);
  //cudaFree(d_B);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_out2);
  cudaFree(d_agg1);
  cudaFree(d_agg2);

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
