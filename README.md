# random-access-bench

Current version based on the HPCC random access benchmark, found at https://github.com/nattoheaven/cuda_randomaccess/blob/master/randomaccess.cu ; refer to http://icl.cs.utk.edu/projectsfiles/hpcc/RandomAccess/ for more info.
Compile:
```
nvcc -o randomaccess randomaccess.cu
```

Run:
```
./randomaccess
```


**OLD**
Compile:
```
nvcc -o indirectCUDA indirectCUDA.cu -ccbin=g++ --compiler-options='-mcmodel=large' -arch=s
m_70 -gencode=arch=compute_70,code=sm_70 -g -G -I /usr/local/cuda-9.1/NVIDIA_CUDA-9.1_Samples/common/inc
```

Run:
```
./indirectCUDA
```

Benchmarks byte random accesses on GPU memory. 

Use https://github.com/cowsintuxedos/random-access-bench/blob/master/indirectCUDA_v4.cu 

Designed to be run on an Amazon Ubuntu Deep Learning AMI running p3.2xlarge, i.e a single Nvidia Tesla V100 GPU.
