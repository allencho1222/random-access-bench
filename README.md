# random-access-bench

Compile:
```
nvcc -o indirectCUDA indirectCUDA.cu -ccbin=g++ --compiler-options='-mcmodel=large' -arch=s
m_70 -gencode=arch=compute_70,code=sm_70 -g -G -I /usr/local/cuda-9.1/NVIDIA_CUDA-9.1_Samples/common/inc
```

Benchmarks byte random accesses on GPU memory. 

Use https://github.com/cowsintuxedos/random-access-bench/blob/master/indirectCUDA.cu 

Designed to be run on an Amazon Ubuntu Deep Learning AMI running p3.2xlarge, i.e a single Nvidia Tesla V100 GPU.
