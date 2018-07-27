# random-access-bench

Compile:
```
nvcc -o indirectTest indirectTest.cu -ccbin=g++ --compiler-options='-mcmodel=large' -arch=s
m_70 -gencode=arch=compute_70,code=sm_70 -g -G -I /usr/local/cuda-9.1/NVIDIA_CUDA-9.1_Samples/common/inc
```
