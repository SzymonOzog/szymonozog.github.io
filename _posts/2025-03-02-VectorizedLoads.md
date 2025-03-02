--- 
title: When and why are vectorized loads faster in CUDA?
date: 2025-03-02
---
Let's start with the most important fact:
[CUDA memory transfers are 128 bytes in size](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis/report/cudaexperiments/sourcelevel/memorytransactions.htm) and each transaction brings one cache line consisting of [4 sectors](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#id28) when we issue a memory load.

This means that if we were to load one float (4 bytes) for each thread in a warp (32 threads) we would be utilizing a full memory transaction 
(32*4 = 128).

Increasing the size of the loaded value to 16 bytes(`float4`) does not change the amount of transactions necessary to get our data into the registers. This can be verified with a following benchmark:

```C
__global__ void copy(int n , float* in, float* out)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = in[i];
  }
}

__global__ void copyf4(int n , float4* in, float4* out)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = in[i];
  }
}
```

Results are confirming what we've just noticed:

![image](https://github.com/user-attachments/assets/b61989c4-24ef-4b9c-8472-51c45e79c555)

Almost no time difference when using a vectorized load than when loading single floats

The situation changes once we switch to a `half` datatype, now we are loading 2 bytes * 32 threads = 64 bytes. This means that only half of our memory transactions will get utilized

```C
__global__ void copy(int n , half* in, half* out)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = in[i];
  }
}

__global__ void copyh2(int n , half2* in, half2* out)
{
  unsigned long i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    out[i] = in[i];
  }
}
```

Running the benchmarks we can see that indeed, vectorizing our loads leads to a much faster memory load(also notice how satisfying the time for half2 is):

![image](https://github.com/user-attachments/assets/9670d6b4-8b26-45d6-8f28-68287c1b9f49)

Since we are in NCU already, we can confirm our assumptions by looking at detailed warp statistics:

For the non vectorized kernel:

![image](https://github.com/user-attachments/assets/dadc4437-f889-44f5-b710-890f4e747be7)

And for the vectorized kernel:

![image](https://github.com/user-attachments/assets/3206cfcb-59bb-46dd-84fc-87249f372002)

We can clearly see that both kernels request the same amount of 32 byte sectors for both kernels. But the vectorized one utilized 4 sectors per requests and makes half of the requests of the non vectorized one.

The situation changes once we start using for loops in our code. Consider the following example:


```C
__global__ void copy_loop(int n , float* in, float* out, int max_size)
{
  unsigned long i = blockIdx.x * blockDim.x;
  for (int idx = i * max_size; idx < (i+blockDim.x)*max_size; idx+=blockDim.x)
  {
      if (idx<n)
      {
        out[idx+threadIdx.x] = in[idx+threadIdx.x];
      }
  }
}

__global__ void copy_loop_float4(int n , float4* in, float4* out, int max_size)
{
  unsigned long i = blockIdx.x * blockDim.x;
  for (int idx = i * max_size; idx < (i+blockDim.x)*max_size; idx+=blockDim.x)
  {
      if (idx<n)
      {
        out[idx+threadIdx.x] = in[idx+threadIdx.x];
      }
  }
}
```

This time our vectorized code is much faster:
![image](https://github.com/user-attachments/assets/f371abe7-8173-475a-b7e6-5d62ad0d2637)

To understand why this happens we need to look at the generated assembly:

![image](https://github.com/user-attachments/assets/fd4c4b98-e86b-4e3a-936d-767c07095e31)

Here we can see that the only difference is that we are now using a 128 bit versions of our load and store instructions. 

So our GPU is running almost the same instructions. But in the case of the vectorized load the amount of times we execute our loop is reduced by 4. This means that we are doing much less integer arithmetic for our indexing operations hence the time savings.

All benchmarks were done on a 4090.
