--- 
title: Writing my own communications library - a worklog of creating Penny Part 2
date: 2025-10-26
---

This is a part two of a worklog on me creating my own communications library, if you didn't read part one you might want to catch up 
[here](https://szymonozog.github.io/posts/2025-09-21-Penny-worklog-1.html). If you've read it or don't care - let's roll.

# Where we left off

Previously we've managed to get quite a good performance with two ring algorithms. One was optimized for throughput and one for latency. We also 
touched on how to think of multi GPU programming when we're going internode. The big gap we have right now against NCCL is when running allreduce across
very small buffers. This part will focus on fixing this, sadly, due to hardware constrains on my side only intranode for this part

<img alt="491993039-737a6a02-661f-4933-9a49-b4c18e9e9cfb" src="https://github.com/user-attachments/assets/e23a3e5e-a354-4241-ba60-b9bfab374225" />

# Giving credit where it belongs

This part will be mostly based on the algorithms that are used in most LLM serving apps, most of what I did learn for this comes from reading the 
[PR](https://github.com/vllm-project/vllm/pull/2192) by Hanzhi Zhou that introduced them to vLLM alongside a PDF that explains the algorithms and the motivation
for using them. I highly recommend reading through both the code and the report as they are very high quality.


# One Shot AllReduce

First algorithm that we will be implementing will be a one shot allreduce. It's the simplest and the least sophisticated algorithm that someone can come up with
and it turns out it's actually quite awesome for very small buffers.

You basically just have all of the pes send data to all other pes and reduce them locally. This is very throughput intensive but again, 
we want this for very small buffers. We send `N_PES*BUFFER_SIZE` worth of data but we only do `N_PES` sends so the latency is very favorable. 

![PennyOneShot-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/61ac4cc4-5269-4a2a-9f6a-2a1c6f071098)


## First try

For the first iteration of the algoritm I decided to just do the simplest thing possible which would be sending all of the data first and then as the data is received we add it to the buffer

```C
template <typename scalar_t>
__global__ void all_reduce_oneshot_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint32_t pe_off = block_size/sizeof(scalar_t);
    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    // send data to all PEs
    for (int send_pe = 0; send_pe<n_pes; send_pe++)
    {
        if (send_pe == pe)
            continue;
        nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                buffer,
                block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, send_pe);
    }
    //Add data to buffer as it comes in
    for (int recv_pe = 0; recv_pe<n_pes; recv_pe++)
    {
        // We have the data from the local PE, we can skip it
        if (recv_pe == pe)
            continue;
        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(signal+recv_pe, NVSHMEM_CMP_EQ, stage);
        __syncthreads();
        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer)[i];
            P dst = reinterpret_cast<P*>(destination + recv_pe*pe_off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer)[i] = res;
        }
    }
}
```
<img alt="Oneshot_1" src="https://github.com/user-attachments/assets/95166e52-558a-460f-9177-16f974cad8e2" />

This already gives us better performance than our ring reductions but there's still a lot of room for improvement, there are two issues right now. 

- We launch only one block, underutiling our GPU
- We wait for the buffers in order. What if data from GPU 7 comes first, and from GPU 0 comes last? Currently we would wait while we can already start adding the data from GPU 7


## Atomic reduction

The second approach tackes this issue, first we declare a global lock that will keep track of wheather our buffer is being used by some other block

```C
__device__ int buffer_lock = 0;
```

One block is responsible for sending the data

``` C

    if (blockIdx.x == pe)
    {
        for (int send_pe = 0; send_pe<n_pes; send_pe++)
        {
            if (send_pe == pe)
                continue;
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                    buffer,
                    block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, send_pe);
        }
        return;
    }
```

And the other blocks wait untill they recieve the data from other PEs and the lock is freed. Afterwards they add to the buffer and release the lock
```C
    int recv_pe = blockIdx.x;
    if (threadIdx.x == 0)
    {
        nvshmem_signal_wait_until(signal+recv_pe, NVSHMEM_CMP_EQ, stage);
        while (atomicCAS(&buffer_lock, 0, 1) != 0) {/*wait*/}
    }

    __syncthreads();

    for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
    {
        P buf = reinterpret_cast<P*>(buffer)[i];
        P dst = reinterpret_cast<P*>(destination + recv_pe*pe_off)[i];
        P res;
        for (int j = 0; j < P::size; j++)
            res.data[j] = float(buf.data[j]) + float(dst.data[j]);
        reinterpret_cast<P*>(buffer)[i] = res;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicExch(&buffer_lock, 0); // Release lock
    }
}

```

Let's benchmark it again.

<img alt="Oneshot_2" src="https://github.com/user-attachments/assets/7a581b30-9ddd-4fd0-b788-08e8a985df93" />

Okay, the performance go better but not by a lot. Currently the bottleneck is that we only do memry transfers using one block, we can split this across other blocks:

```C
if (blockIdx.x != pe)
{
        nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                buffer,
                block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, blockIdx.x);
}
else
{
    return;
}
```

<img alt="Oneshot_3" src="https://github.com/user-attachments/assets/a0319464-08bb-45fe-945d-739cb49e3f6c" />


Now we're getting fast. But we can still do better than this


## Parallel reduction

The next thing I tried was doing a reduction in parallel, currently it takes `N_PES` steps as we need to wait untill the previous node releases the lock. What we can do is a tree recuction that will change it to `log2(N_PES)` steps. 

I could scribe 16x16 words but it's easier to show an image of this:

<img alt="Untitled-2025-09-19-1412" src="https://github.com/user-attachments/assets/186b37fa-b7a9-4900-951f-36eaf2406362" />


First we need to change the lock, right now it signals when the buffers are ready to be read again

```C
__device__ int buffer_lock[4] = {0};
```

I define a helper reduciton function that takes two input buffers that we do reduction across and a destination buffer we save the results to
```C


    auto reduce = [&](scalar_t* s1_p, scalar_t* s2_p, scalar_t* dst)
    {
        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P src1 = reinterpret_cast<P*>(s1_p)[i];
            P src2 = reinterpret_cast<P*>(s2_p)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(src1.data[j]) + float(src2.data[j]);
            reinterpret_cast<P*>(dst)[i] = res;
        }
    };
```

We can now perform the three steps that we need to reduce across 8 GPUS, each step the blocks that don't do the job mark the buffer as ready and exit while the other blocks wait for the data from both inputs to be valid and perform the reduction

```C

    int off = 4;
    if (blockIdx.x >= off)
    {
        return;
    }

    int recv_pe0 = blockIdx.x;
    int recv_pe1 = blockIdx.x + off;

    if (threadIdx.x == 0 && recv_pe0 != pe)
    {
        nvshmem_signal_wait_until(signal+recv_pe0, NVSHMEM_CMP_EQ, stage);
    }
    if (threadIdx.x == 1 && recv_pe1 != pe)
    {
        nvshmem_signal_wait_until(signal+recv_pe1, NVSHMEM_CMP_EQ, stage);
    }

    __syncthreads();
    reduce(recv_pe0 == pe ? buffer : destination + recv_pe0*pe_off,
            recv_pe1 == pe ? buffer : destination + recv_pe1*pe_off,
            destination + recv_pe0*pe_off);
    __syncthreads();

    off /= 2;
    if (blockIdx.x >= off)
    {
        if (threadIdx.x == 0) { atomicAdd(&buffer_lock[blockIdx.x], 1); }
        return;
    }
    if(threadIdx.x == 0)
    {
        while(atomicCAS(&buffer_lock[blockIdx.x+off], 1, 0) != 1) { }
    }
    __syncthreads();
    recv_pe1 = blockIdx.x + off;
    reduce(destination + recv_pe0*pe_off,
            destination + recv_pe1*pe_off,
            destination + recv_pe0*pe_off);
    __syncthreads();

    off /= 2;
    if (blockIdx.x >= off)
    {
        if (threadIdx.x == 0) { atomicAdd(&buffer_lock[blockIdx.x], 1); }
        return;
    }
    if(threadIdx.x == 0)
    {
        while(atomicCAS(&buffer_lock[blockIdx.x+off], 1, 0) != 1) {}
        nvshmem_quiet();
    }
    __syncthreads();
    // Now we save to the buffer
    recv_pe1 = blockIdx.x + off;
    reduce(destination + recv_pe0*pe_off,
            destination + recv_pe1*pe_off,
            buffer);
}
```

<img alt="Oneshot_4" src="https://github.com/user-attachments/assets/8ef0b8c6-d904-4d91-81e3-5340079e1bc6" />


With this we're getting even better performance than before

## Maybe all on one go will be faster

At this point the performance was good but I had a weird suspition, we keep writing to global memory a lot when doing our reduction. What if we just did this all in one go? 
Just wait untill all of the data arrives from memory and just read and write to global memory once? Let's try it out:

```C
template <typename scalar_t, int N_PES = 8>
__global__ void all_reduce_oneshot_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint32_t pe_off = block_size/sizeof(scalar_t);

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    if (blockIdx.x != pe && blockIdx.y == 0)
    {
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off,
                    buffer,
                    block_size, signal+pe, stage, NVSHMEM_SIGNAL_SET, blockIdx.x);
    }

    //wait until we receive all of the data
    for(int tid = 0; tid<N_PES; tid++)
    {
        if (threadIdx.x == tid && tid != pe)
        {
            nvshmem_signal_wait_until(signal+tid, NVSHMEM_CMP_EQ, stage);
        }
    }

    __syncthreads();
    // Right now each block parallelizes across buffer size instead of pes
    const uint32_t reduce_size = block_size/(N_PES*gridDim.y);
    const uint32_t reduce_off = (blockIdx.y*gridDim.x + blockIdx.x)*reduce_size/sizeof(scalar_t);

    //reduce and write
    for (int i = threadIdx.x; i < reduce_size/(sizeof(P)); i += blockDim.x)
    {
        P res = reinterpret_cast<P*>(buffer + reduce_off)[i];
        for (int recv_pe = 0; recv_pe < N_PES; recv_pe++)
        {
            if(recv_pe == pe)
                continue;
            P src = reinterpret_cast<P*>(destination + recv_pe*pe_off + reduce_off)[i];
            for (int j = 0; j < P::size; j++)
            {
                res.data[j] += float(src.data[j]);
            }
        }
        reinterpret_cast<P*>(buffer +  reduce_off)[i] = res;
    }
}
```

<img alt="Oneshot_5" src="https://github.com/user-attachments/assets/c52acd61-021f-4272-b485-1cc257b8e0f6" />

Okay wow, it's actually much better, turns out that the simplest thing outperformed all of my sophisticated parallel reductions. 

For the final touch I decided to add a little bit of search. With gridDim.z we can controll how many packets we divide our block into and with gridDim.y we can controll how many blocks perform the reduction

```C
template <typename scalar_t, int N_PES = 8>
__global__ void all_reduce_oneshot_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, scalar_t* __restrict__ output, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint32_t pe_off = block_size/sizeof(scalar_t);
    const uint32_t off = blockIdx.z * pe_off;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    if (blockIdx.x != pe && blockIdx.y == 0)
    {
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off + off*N_PES,
                    buffer + off,
                    block_size, signal+pe + blockIdx.z*N_PES, stage, NVSHMEM_SIGNAL_SET, blockIdx.x);
    }

    for(int tid = 0; tid<N_PES; tid++)
    {
        if (threadIdx.x == tid && tid != pe)
        {
            nvshmem_signal_wait_until(signal+tid + blockIdx.z*N_PES, NVSHMEM_CMP_EQ, stage);
        }
    }

    __syncthreads();
    const uint32_t reduce_size = block_size/(N_PES*gridDim.y);
    const uint32_t reduce_off = (blockIdx.y*gridDim.x + blockIdx.x)*reduce_size/sizeof(scalar_t);

    for (int i = threadIdx.x; i < reduce_size/(sizeof(P)); i += blockDim.x)
    {
        P res = reinterpret_cast<P*>(buffer + reduce_off + off)[i];
        for (int recv_pe = 0; recv_pe < N_PES; recv_pe++)
        {
            if(recv_pe == pe)
                continue;
            P src = reinterpret_cast<P*>(destination + recv_pe*pe_off + reduce_off + off*N_PES)[i];
            for (int j = 0; j < P::size; j++)
            {
                res.data[j] += float(src.data[j]);
            }
        }
        reinterpret_cast<P*>(output + reduce_off + off)[i] = res;
    }
}

```

Let's test it out:

<img alt="Oneshot_6" src="https://github.com/user-attachments/assets/c1f27af4-1b42-4fb2-825b-8d8e51a874b3" />


A small improvement but an improvement nevetheless


# Two Shot AllReduce

This is a more bandwidth optimized version of our One Shot algorithm, right now instead of sending a full buffer to each pe, we first divide it into `N_PES` chunks
and send it across our node, afterwards each PE performs a reduction and broadcasts the result to all other pes.

![PennyTwoShot-ezgif com-video-to-gif-converter](https://github.com/user-attachments/assets/2535bd1d-f9a5-4195-875e-7d811f013693)

It's very simillar to the code we had for the oneshot but with a broadcast phade added now. It also implements the search patterns across packet size and block size

```C
template <typename scalar_t, int N_PES=8>
__global__ void all_reduce_twoshot_kernel(scalar_t* __restrict__ destination, scalar_t*  buffer, scalar_t* __restrict__ output, uint64_t* signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    const uint32_t block_size = blockDim.x * packet_size;

    const uint32_t pe_off = (block_size*gridDim.z)/sizeof(scalar_t);
    const uint32_t off = blockIdx.z * block_size/sizeof(scalar_t);

    uint32_t write_chunk = blockIdx.x;

    // Send all the data to other PEs
    if (write_chunk != pe && blockIdx.y == 0)
    {
            nvshmemx_putmem_signal_nbi_block(destination + pe*pe_off + off,
                    buffer + write_chunk*pe_off + off,
                    block_size, signal+pe + 2*blockIdx.z*N_PES, stage, NVSHMEM_SIGNAL_SET, write_chunk);
    }

    // Wait for the data from all PEs to be ready
    for(int tid = 0; tid<N_PES; tid++)
    {
        if (threadIdx.x == tid && tid != pe)
        {
            nvshmem_signal_wait_until(signal+tid + 2*blockIdx.z*N_PES, NVSHMEM_CMP_EQ, stage);
        }
    }

    // Perform a reduction across the chunk belonging to this block
    __syncthreads();
    const uint32_t reduce_size = block_size/(N_PES*gridDim.y);
    const uint32_t reduce_off = (blockIdx.y*gridDim.x + blockIdx.x)*reduce_size/sizeof(scalar_t);

    for (int i = threadIdx.x; i < reduce_size/(sizeof(P)); i += blockDim.x)
    {
        P res = reinterpret_cast<P*>(buffer + pe*pe_off + off + reduce_off)[i];
        for (int recv_pe = 0; recv_pe < N_PES; recv_pe++)
        {
            if(recv_pe == pe)
                continue;
            P src = reinterpret_cast<P*>(destination + recv_pe*pe_off + off + reduce_off)[i];
            for (int j = 0; j < P::size; j++)
            {
                res.data[j] += float(src.data[j]);
            }
        }
        reinterpret_cast<P*>(output + pe*pe_off + off + reduce_off)[i] = res;
    }

    // wait till the writes are visible to other blocks
    __threadfence_system();
    __syncthreads();
    // Mark chunk as ready
    if (threadIdx.x == 0)
    {
        nvshmemx_signal_op(signal+n_pes+pe + 2*blockIdx.z*N_PES, 1, NVSHMEM_SIGNAL_ADD, pe);
    }
    if (write_chunk == pe)
    {
        return;
    }

    // Wait till all chunks are ready
    if (threadIdx.x == 0)
    {

        nvshmem_signal_wait_until(signal+n_pes+pe + 2*blockIdx.z*N_PES, NVSHMEM_CMP_EQ, stage*N_PES*gridDim.y);
    }

    __syncthreads();

    // Send data to other PEs
    if(blockIdx.y == 0)
    {
        nvshmemx_putmem_signal_nbi_block(destination + (n_pes+pe)*pe_off + off,
                output + pe*pe_off + off,
                block_size, signal+n_pes+pe + 2*blockIdx.z*N_PES, stage, NVSHMEM_SIGNAL_SET, write_chunk);
    }

    // Wait till we receive data from other PE
    if (threadIdx.x == 0)
        nvshmem_signal_wait_until(signal+n_pes+write_chunk + 2*blockIdx.z*N_PES, NVSHMEM_CMP_EQ, stage);
    __syncthreads();

    const uint32_t write_size = block_size/gridDim.y;
    const uint32_t write_off = (blockIdx.y*write_size)/sizeof(scalar_t);

    // Write it to our output
    for (int i = threadIdx.x; i < write_size/(sizeof(P)); i += blockDim.x)
    {
        reinterpret_cast<P*>(output + write_chunk*pe_off + off + write_off)[i] =
            reinterpret_cast<P*>(destination + (n_pes+write_chunk)*pe_off + off + write_off)[i];
    }
}
```

With this we can get better than NCCL performance for medium sized buffers as well 


# Comparison of all algoritms

For completeness let's visualize all of the algorithms that we've implemented so far. I started this post mentioning that all of the algorithms here are based on the custom allreduce from inside vLLM, it would be fair to compare against it as well

<img alt="cumulative2" src="https://github.com/user-attachments/assets/04545b7f-5ba3-4c95-8e56-26023faff024" />

The good news is that we're now outperforming NCCL on every buffer size that matters for LLM inference. 
The bad news is we're getting slightly worse performance on very small buffers compared to the vLLM version. The reason is very simple and a limitation of our current approach that is using NVSHMEM. 
Since it relies on symmetric heap, we do need to move the data in/out of the destination buffer. While this is negligable for bigger buffers in a latency bound scenario
it gives us just a slight overhead. I though about redoing this with IPC buffers just to close that scenario but I decided not to, the kernels are there and well documented
so be sure to read the PR I linked before if you want to see how it can be done with a different abstraction. 

Another thing worth noting is that the simple ring turned out useless. I'll skip it in all further parts unless I find out how to make it better

Untill next time!
