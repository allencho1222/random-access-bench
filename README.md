# random-access-bench

Compile:
```
nvcc -o indirectTest indirectTest.cu -ccbin=g++ --compiler-options='-mcmodel=large' -arch=s
m_70 -gencode=arch=compute_70,code=sm_70 -g -G -I /usr/local/cuda-9.1/NVIDIA_CUDA-9.1_Samples/common/inc
```


Designed to be run on an Amazon Ubuntu Deep Learning AMI running p3.2xlarge, i.e a single Nvidia Tesla V100 GPU.
