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
	rows = 1U << 18,
  array = 1U << 18,
	groups = 1U << 10,
	segment_bits = 12,
  segments = array / (1U << segment_bits)
};

struct Row {
	unsigned int measure;
	unsigned int group;
};

struct String {
  char str[128];
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

// declare fnv hash functions and vars
#define FNV_PRIME	0x01000193

#define fnv(x,y) ((x) * FNV_PRIME ^(y))

__device__ uint4 fnv4(uint4 a, uint4 b)
{
	uint4 c;
	c.x = a.x * FNV_PRIME ^ b.x;
	c.y = a.y * FNV_PRIME ^ b.y;
	c.z = a.z * FNV_PRIME ^ b.z;
	c.w = a.w * FNV_PRIME ^ b.w;
	return c;
}

__device__ uint32_t fnv_reduce(uint4 v)
{
	return fnv(fnv(fnv(v.x, v.y), v.z), v.w);
}

// generate random 128-byte string
__device__ void random_string(char *s)
{
  int tId = threadIdx.x + (blockIdx.x * blockDim.x);
  curandState state;
  curand_init((unsigned long long)clock() + tId, 0, 0, &state);

  char alphanum[] = {
    'a','b','c','d','e','f','g','h','i','j','k','l','m',
    'n','o','p','q','r','s','t','u','v','w','x','y','z',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    '0','1','2','3','4','5','6','7','8','9',
    '!','@','#','$','%','^','&','*','(',')','-','_','+','=',
    '<','>',',','.','/','?',';',':','[',']','{','}'
  };

  for (int i = 0; i < 128; ++i) {
    s[i] = alphanum[static_cast<int>(curand_uniform(&state) * (unsigned)(sizeof(alphanum) - 1))];
    printf("%c", s[i]);
  }

  s[128] = 0;
  printf("\n");
}

__device__ struct String dd_A[array];
__device__ unsigned int dd_in[rows];
__device__ struct String dd_out[rows];

// initialize the GPU arrays
__global__ void d_init()
{
    printf("Initializing data structures.\n");
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);
    curandState state;
    curand_init((unsigned long long)clock() + tId, 0, 0, &state);
    printf("Size of word: %lu bytes\n", (unsigned long)sizeof(dd_A[0].str));

    // Random fill indirection array A
    unsigned int i;
    printf("Randomly filling array A.\n");
    for (i = 0; i < array; i++) {
        //d_A[i].measure = curand_uniform(&state) * array;
        //d_A[i].group = curand_uniform(&state) * groups;
        printf("dd_A[%d] - ",i);
        random_string(dd_A[i].str);

        //printf("d_A[%d] - %d\n",i,d_A[i].measure);
    }

    // Random fill input
    printf("Random filling input array.\n");
    for (i = 0; i < rows; i++) {
        dd_in[i] = curand_uniform(&state) * array;
        //random_string(dd_A[i].str);

        //printf("dd_in[%d] - %d\n",i,dd_in[i]);
    }
    printf("Successfully initialized input array.\n");
}

__global__ void d_bench()
{
    // ikimasu
    //printf("Initializing benchmarks.\n");

  /*
    // Gather rows
    unsigned i;
    printf("Beginning gather\n");
	for (i = 0; i < rows; i++) {
		d_out[i] = d_A[d_in[i]];
  }*/
  
  // bench 128-byte reads
  unsigned i;
  for (i = 0; i < rows; i++) {
    for (int j = 0; j < 128; j++) {
      dd_out[i].str[j] = dd_A[dd_in[i]].str[j];
      //printf("%c",dd_out[i].str[j]);
    }
    //dd_out[i] = dd_A[dd_in[i]];
    //printf("\n");
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
  d_init << <1, 1>> >();


  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  cudaEventRecord(begin);
  cudaEventSynchronize(begin);

  //d_bench << <grid, thread >> >();
  printf("Beginning GPU benchmark.\n");
  d_bench << <1, 1 >> >();

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
