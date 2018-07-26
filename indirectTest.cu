//
// Author     :  matto@xilinx 14JAN2018, alai@xilinx 25JULY2018
// Filename   :  indirectTest.cu
// Description:  Cuda random access benchmark example based on indirect.c by gswart/skchavan@oracle
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <utime.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <curand.h>
#include <curand_kernel.h>

#define DEBUG
#define CPU_BENCH
#define NOCUDA

#define MEM_LOGN 28
//#define GATHER2

#define FULLMEM
//#define VERIF

#ifdef FULLMEM
enum {
	rows = 1U << 10, // above 18 for rows or arrays causes segfault
        array = 1U << 10,
        rows_test = 1U << 26,
        array_test = 1U << 26,
	groups = 1U << 18,
	segment_bits = 18,
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
//void *rows2;
//cudaError_t error = cudaMalloc(&rows2, (1U << 31));
//__device__ struct Row d_A[array];
__device__ struct Row d_A[array_test];
//__device__ struct Row *d_A;
//  cudaError_t error = cudaMalloc((void**) &d_A, (array_test*sizeof(struct Row)) );
//__device__ unsigned int d_in[rows];
__device__ unsigned int d_in[rows_test];
//__device__ struct Row d_out[rows];
__device__ struct Row d_out[rows_test];
__device__ unsigned long long d_agg1[groups];
__device__ unsigned long long d_agg2[groups];
//__device__ struct Row d_out2[rows];
__device__ struct Row d_out2[rows_test];
//__device__ struct Row * d_B[segments];

__global__ void d_bench()
{
	// ikimasu
	//struct Row A[array];

        //unsigned int in[rows];
        //struct Row out[rows];
        //unsigned long long agg1[groups];
        //unsigned long long agg2[groups];

        //struct Row out2[rows];
        //struct Row * B[segments];

	int tId = threadIdx.x + (blockIdx.x * blockDim.x);
	curandState state;
	curand_init((unsigned long long)clock() + tId, 0, 0, &state);

  	printf("Initializing data structures.\n");

  	// Random fill indirection array A
  	unsigned int i;
	printf("Random filling A.\n");
  	for (i = 0; i < array_test; i++) {
          //d_A[i].measure = rand() % array_test;
          //d_A[i].group = rand() % groups;
	  d_A[i].measure = curand_uniform(&state) * array_test;
          d_A[i].group = curand_uniform(&state) * groups;
	  //printf("%d\n",d_A[i].measure);
	  //printf("%d\n",d_A[i].group);

	  //d_A[i].measure = i;
          //d_A[i].group = i % groups;
	  printf("d_A[%d] - %d\n",i,d_A[i].measure);
          //printf("%d\n",d_A[i].group);
  	}

  	// Fill segmented array B
  	/**for (i = 1; i <= segments; i++) {
          d_B[i] = &(d_A[i * (1U << segment_bits)]);
  	}**/

  	// Random fill input
	printf("Random filling input.\n");
  	for (i = 0; i < rows_test; i++) {
          //d_in[i] = rand() % array_test;
	  d_in[i] = curand_uniform(&state) * rows_test;
	  //printf("%d\n",d_in[i]);

	  //d_in[i] = i;
	  printf("d_in[%d] - %d\n",i,d_in[i]);
	}

  	// Zero aggregates
  	for (i = 0; i < groups; i++) {
          d_agg1[i] = 0;
          d_agg2[i] = 0;
  	}
	//unsigned int i;

	// Gather rows
	for (i = 0; i < rows; i++) {
		d_out[i] = d_A[d_in[i]];
	}

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
	}

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


static void
init()
{
	struct Row A[array];

	unsigned int in[rows];
	struct Row out[rows];
	unsigned long long agg1[groups];
	unsigned long long agg2[groups];

	struct Row out2[rows];
	struct Row * B[segments];

  printf("Initializing data structures.\n");

  // Random fill indirection array A
  unsigned int i;
  for (i = 0; i < array; i++) {
	  A[i].measure = rand() % array;
	  A[i].group = rand() % groups;
  }

  // Fill segmented array B
  for (i = 0; i < segments; i++) {
	  B[i] = &(A[i * (1U << segment_bits)]);
  }

  // Random fill input
  for (i = 0; i < rows; i++)
	  in[i] = rand() % array;

  // Zero aggregates
  for (i = 0; i < groups; i++) {
	  agg1[i] = 0;
	  agg2[i] = 0;
  }

#ifdef NOCUDA
  // ikimasu
  //struct Row *Acpy = new struct Row[array];
  //std::copy(A, A+array, Acpy);
  //for (i = 0; i < array; i++) {
  //  printf("woohoo");
  //  Acpy[i] = A[i];
    //d_A[i] = A[i];
    //cudaMemcpyToSymbol(d_A[i], A[i], (array*sizeof(A[i])));
  //}
  //checkCudaErrors(cudaMemcpyToSymbol(d_A, &Acpy, sizeof(Acpy)));

  //checkCudaErrors(cudaMemcpyToSymbol(d_A, A, sizeof(A)));
  //checkCudaErrors(cudaMemcpyToSymbol(d_B, B, sizeof(B)));
  //checkCudaErrors(cudaMemcpyToSymbol(d_in, in, sizeof(in)));
  //checkCudaErrors(cudaMemcpyToSymbol(d_out, out, sizeof(out)));
  //checkCudaErrors(cudaMemcpyToSymbol(d_out2, out2, sizeof(out2)));
  //checkCudaErrors(cudaMemcpyToSymbol(d_agg1, agg1, sizeof(agg1)));
  //checkCudaErrors(cudaMemcpyToSymbol(d_agg2, agg2, sizeof(agg2)));
#endif // !1


#ifdef CPU_BENCH

  // Gather rows
  for (i = 0; i < rows; i++) {
	  out[i] = A[in[i]];
  }

  // Indirect Gather rows
  for (i = 0; i < rows; i++) {
	  out[i] = A[A[in[i]].measure];
  }

  // Fused gather group
  for (i = 0; i < rows; i++) {
	  agg2[A[in[i]].group] += A[in[i]].measure;
#ifdef DEBUG
	  printf("CPU:  agg2[A[in[i]].group]  = %d\n", agg2[A[in[i]].group]);
#endif // DEBUG  
#endif // CPU_BENCH
  }

}

int
main(int argc, char *argv[])
{
  
  init();

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

  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);
  cudaEventSynchronize(begin);

  //d_bench << <grid, thread >> >();
  d_bench << <1, 1 >> >();

  cudaEventRecord(end);
  cudaEventSynchronize(end);

  float ms;
  cudaEventElapsedTime(&ms, begin, end);
  cudaEventDestroy(end);
  cudaEventDestroy(begin);
  double time = ms * 1.0e-3;
  printf("Elapsed time = %.6f seconds.\n", time);

#endif // !1

#ifdef VERIF
  //d_check << <grid, thread >> >(n, d_t);
  cpu_bench();
#endif // VERIF

#ifdef NOCUDA
  //cudaFree(rows2);

  cudaFree(d_A);
  //cudaFree(d_B);
  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_out2);
  cudaFree(d_agg1);
  cudaFree(d_agg2);

#endif // !1

  return 0;
}
