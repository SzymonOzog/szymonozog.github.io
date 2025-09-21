--- 
title: Writing my own communications library - a worklog of creating Penny part 1
date: 2025-09-21
---
## Motivation
"What I cannot create I do not understand" - This is why I started Penny, my own version of NCCL.

A goal of mine would be to be able to swap Penny and NCCL in an LLM serving framework and see close to no performance degradation.
Choosing LLM inference makes things simpler as it almost only relies on AllReduce so this is the first algorithm that I'll try to implement.

As an implementation tool for it I chose NVSHMEM, this is a communication library from NVIDIA that's based on OpenSHMEM standard. The important part is that as opposed to NCCL it has a device API,
meaning that we can send data from one GPU to another while executing the kernel. Imagine the possibilities. It takes away the fun of implementing all of
the low level communication stuff and gives us higher level primitives that we can work with to send data between our GPUs, but as 
much as I'd love to get to know this stuff I'm afraid that implementing this myself would be too big in scope and the project would
end up on the graveyard of my private github unfinished projects. I'll leave this as a sidequest for later.

This will be the first part of a worklog on it, showing my progress. That being said, they will evolve over time as I'm learning new things about GPU communication. 
Obviously I cannot write NCCL on my own so there are tradeoffs to be made. I'm not gonna optimize that much for reducing the usage of GPU resources(SMs and memory) and will focus on correctness and speed.

That being said the first part of the worklog will have four sections: 

- An introduction to how GPU to GPU communication works
- Get NVSHMEM set up, investigate the API, create a simple example and measure our bandwidth.
- Write an efficient AllReduce on a single node
- Scale our algorithm to multiple nodes

## GPU Communication 101

First of all, we need to take a look at how communication works on GPU nodes. One DGX node consists of 8 GPUs, each one connected to a number of Network Interface Cards(NICs), they allow us to communicate
with network switches to send data outwards. Inside the nodes, all GPUs are interconnected with NVLink.

I've visualized this on the image below, but reduced this to 4 GPUs per node not to clutter this too much. I think you get the idea.

<img alt="SETUP" src="https://github.com/user-attachments/assets/bcea6c98-af59-465e-8d6c-e76131005928" />

To check what NICs there are available we can run `nvidia-smi topo -m`

```
        GPU0    GPU1    GPU2    GPU3    GPU4    GPU5    GPU6    GPU7    NIC0    NIC1    NIC2    NIC3    NIC4    NIC5    NIC6    NIC7    NIC8    NIC9    NIC10   NIC11   NIC12   CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV18    NV18    NV18    NV18    NV18    NV18    NV18    PIX     PXB     NODE    NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-55,112-167    0               N/A
GPU1    NV18     X      NV18    NV18    NV18    NV18    NV18    NV18    PXB     PIX     NODE    NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-55,112-167    0               N/A
GPU2    NV18    NV18     X      NV18    NV18    NV18    NV18    NV18    NODE    NODE    PIX     PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-55,112-167    0               N/A
GPU3    NV18    NV18    NV18     X      NV18    NV18    NV18    NV18    NODE    NODE    PXB     PIX     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     0-55,112-167    0               N/A
GPU4    NV18    NV18    NV18    NV18     X      NV18    NV18    NV18    SYS     SYS     SYS     SYS     PIX     PXB     NODE    NODE    NODE    NODE    NODE    NODE    NODE    56-111,168-223  1               N/A
GPU5    NV18    NV18    NV18    NV18    NV18     X      NV18    NV18    SYS     SYS     SYS     SYS     PXB     PIX     NODE    NODE    NODE    NODE    NODE    NODE    NODE    56-111,168-223  1               N/A
GPU6    NV18    NV18    NV18    NV18    NV18    NV18     X      NV18    SYS     SYS     SYS     SYS     NODE    NODE    PIX     PXB     NODE    NODE    NODE    NODE    NODE    56-111,168-223  1               N/A
GPU7    NV18    NV18    NV18    NV18    NV18    NV18    NV18     X      SYS     SYS     SYS     SYS     NODE    NODE    PXB     PIX     NODE    NODE    NODE    NODE    NODE    56-111,168-223  1               N/A
NIC0    PIX     PXB     NODE    NODE    SYS     SYS     SYS     SYS      X      PXB     NODE    NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC1    PXB     PIX     NODE    NODE    SYS     SYS     SYS     SYS     PXB      X      NODE    NODE    SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC2    NODE    NODE    PIX     PXB     SYS     SYS     SYS     SYS     NODE    NODE     X      PXB     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC3    NODE    NODE    PXB     PIX     SYS     SYS     SYS     SYS     NODE    NODE    PXB      X      SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS     SYS
NIC4    SYS     SYS     SYS     SYS     PIX     PXB     NODE    NODE    SYS     SYS     SYS     SYS      X      PXB     NODE    NODE    NODE    NODE    NODE    NODE    NODE
NIC5    SYS     SYS     SYS     SYS     PXB     PIX     NODE    NODE    SYS     SYS     SYS     SYS     PXB      X      NODE    NODE    NODE    NODE    NODE    NODE    NODE
NIC6    SYS     SYS     SYS     SYS     NODE    NODE    PIX     PXB     SYS     SYS     SYS     SYS     NODE    NODE     X      PXB     NODE    NODE    NODE    NODE    NODE
NIC7    SYS     SYS     SYS     SYS     NODE    NODE    PXB     PIX     SYS     SYS     SYS     SYS     NODE    NODE    PXB      X      NODE    NODE    NODE    NODE    NODE
NIC8    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE     X      PIX     PIX     PIX     PIX
NIC9    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    PIX      X      PIX     PIX     PIX
NIC10   SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    PIX     PIX      X      PIX     PIX
NIC11   SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    PIX     PIX     PIX      X      PIX
NIC12   SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    SYS     SYS     SYS     SYS     NODE    NODE    NODE    NODE    PIX     PIX     PIX     PIX      X

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

NIC Legend:

  NIC0: mlx5_ib0
  NIC1: mlx5_ib1
  NIC2: mlx5_ib2
  NIC3: mlx5_ib3
  NIC4: mlx5_ib4
  NIC5: mlx5_ib5
  NIC6: mlx5_ib6
  NIC7: mlx5_ib7
  NIC8: mlx5_eth0
  NIC9: mlx5_eth1
  NIC10: mlx5_eth2
  NIC11: mlx5_eth3
  NIC12: mlx5_eth4

```

This shows us that we have 12 NICs available 4 of them being Ethernet and 8 InfiniBand. For internode communication we'll only use InfiniBand NICs as they are much faster. We can check the speed of each NIC by examining with `ibstatus`

```
Infiniband device 'mlx5_eth4' port 1 status:
        rate:            25 Gb/sec (1X EDR)
        link_layer:      Ethernet

Infiniband device 'mlx5_ib0' port 1 status:
        rate:            400 Gb/sec (4X NDR)
        link_layer:      InfiniBand

```

You can see why we don't want to use Ethernet for communication.

Let's now get into the programming part.


## NVSHMEM


### Library overview

Based on the OpenSHMEM standard, NVSHMEM exposes a couple of simple primitives for programming on multiple GPUs. We can `get` the data from the memory of another GPU or `put` the data 
on the memory of another GPU. This is based on an idea of a symmetric heap. It takes 2 assumptions:
- All processes allocated the same sized buffers
- All allocations have the same offset in memory

This is pretty neat because those assumptions save us from a lot of trouble of mapping the received data to the desired location. This also gives us a few constrains:
- All buffers we write to need to be allocated using `nvshmem_malloc`
- All buffers we write from must be pre registered with `nvshmemx_register_buffer`

Quick side note on the naming conventions. Every time that a function is prefixed with `nvshmem` it's based on an equivalent in the OpenSHMEM standard, if it's prefixed with `nvshmemx`, it's an extension to the standard

### Initialization

Before we start exchanging the data, our processes need to be aware of each other. The default methods for initializing NVSHMEM are MPI or their own launcher called Hydra. I don't want to rely on any of those
since in the end the API will need to be compatible with an LLM serving framework that spawns its own processes. 

Fortunately there is a third way that's surprisingly undocumented looking at how it's the most flexible one. We can initialize using a UUID, it's quite simple. On the host process we can take our unique NVSHMEM UUID:

```C
pybind11::bytearray get_nvshmem_unique_id() 
{
    nvshmemx_uniqueid_t unique_id;
    nvshmemx_get_uniqueid(&unique_id);
    std::vector<uint8_t> result(sizeof(nvshmemx_uniqueid_t));
    std::memcpy(result.data(), &unique_id, sizeof(nvshmemx_uniqueid_t));
    return {reinterpret_cast<const char*>(result.data()), result.size()};
}
```

Then we can use NCCL(cheater!) to synchronize our UUID across processes and initialize all of them with the same attributes:

```py
def initialize_distributed():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    nnodes = int(os.getenv("NNODES"))
    local_size = world_size//nnodes
    local_rank = dist.get_rank() % local_size

    torch.cuda.set_device(local_rank)
    nvshmem_uid = penny_cpp.get_unique_id()

    nvshmem_uids = [None, ] * world_size
    dist.all_gather_object(nvshmem_uids, nvshmem_uid)
    penny_cpp.init_with_uid(nvshmem_uids[0], dist.get_rank(), world_size)
```

```C
void init_with_uid(pybind11::bytearray uid_py, int rank, int world_size)
{
    auto uid_str = uid_py.cast<std::string>();

    nvshmemx_uniqueid_t uid;
    std::memcpy(&uid, uid_str.c_str(), sizeof(nvshmemx_uniqueid_t));
    nvshmemx_init_attr_t attr;
    nvshmemx_set_attr_uniqueid_args(rank, world_size, &uid, &attr);
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}
```

### Sending the data

For transferring the data we can do it 2 ways:

The `put` way:
- GPU0 calls `put` to send the data to GPU1
- GPU1 waits for the signal that the data is ready
- Both GPUS are free to do whatever with the data 

The `get` way:
- GPU1 calls `get` to retrieve the data from GPU0
- GPU0 waits for the signal that the data has been read
- Both GPUS are free to do whatever with the data 

I'll got with the `put` way as it seems more natural to me but AFAIK both ways achieve the same speed(it might be actually a fun exercise to rewrite all of the algorithms using `get`)

There are a lot of versions of `put` that NVSHMEM exposes, from the NVSHMEM standard

`nvshmem_DATATYPE_put`

where `DATATYPE` tells us which data to use(eg. `float`/`int`) and we can specify how many values we will pass

There is also a wildcarded version `nvshmem_putmem` that allows us to send any datatype of any size, we just have to specify the amount of data transferred. I like this idea so that's why I'll go with this version


NVSHMEM also extends the standard with:
- `nvshmemx_putmem_warp`
- `nvshmemx_putmem_block`

Those align with the CUDA programming model and give us a tradeoff between throughput and latency.
- `putmem` will just use a single thread to load the data from device
- `putmem_warp` will use a whole warp to load the data from device memory 
- `putmem_block` will use a whole block to load the data from device memory 

Warp and Block versions will be faster but will use more resources and will call `__syncwarp()`/`__syncthreads()` internally. For our case we're not gonna need the resources anyway, we'll go with the `block` version


### Exchange kernel

As a first learning exercise on how to use NVSHMEM I stated with a simple exchange kernel, basically GPU A swaps all of the contents
of it's buffer with GPU B

In NCCL the equivalent would be:

```py
ops = [dist.P2POp(dist.isend, data, src),
       dist.P2POp(dist.irecv, data_r, src)]
# GPU 0 communicates with GPU 1 and batch_isend_irecv requires matching order of sends and receives
if rank%2:
    ops = list(reversed(ops))
dist.batch_isend_irecv(ops)
torch.cuda.synchronize()
```

The first way that I got it to working in NVSHMEM is with the following pattern:

- Initialize symmetric memory with `nvshemem_malloc`
- Register buffer that we'll be exchanging 
- Call kernel that puts memory on the symmetric heap of our second node
- Synchronize all participating peers to make sure that it's safe to read from `destination` and write to `buffer`
- Copy the data from the symmetric heap to the buffer
- Cleanup

In CUDA code it looks like this:


```C
template <typename scalar_t>
__global__ void exchange(scalar_t* destination, scalar_t* buffer, int peer, int packet_size) 
{
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint64_t block_size = blockDim.x * packet_size;
    nvshmemx_putmem_block(destination + off, buffer + off, block_size, peer);
}

void exchange(torch::Tensor& buffer, int packet_size, int block_size, int peer) 
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    half *destination = (half *) nvshmem_malloc(buffer.numel() * sizeof(half));
    nvshmemx_buffer_register(buffer.data_ptr(), buffer.numel() * sizeof(half));
    
    const uint32_t grid_size = std::ceil(buffer.numel()*sizeof(half) / float(packet_size*block_size));

    exchange<<<grid_size, block_size, 0, stream>>>(destination,
            static_cast<half*>(buffer.data_ptr()),
            peer,
            packet_size);

    nvshmemx_barrier_all_on_stream(stream);

    cudaMemcpyAsync(buffer.data_ptr(), (void*)destination, buffer.numel() * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    cudaStreamSynchronize(stream);

    nvshmemx_buffer_unregister(buffer.data_ptr());
    nvshmem_free(destination);
}
```

`packet_size` in the code is the size in Bytes that a single thread sends, and `block_size` is the amount of threads that work together to 
call our `putmem` function.

To find a good configuration I just ran a sweep across sensible outputs. For intranode we're getting 733 GB/s and for internode we're getting 87 GB/s
Both are very close to the max bandwidth of the transport layers but for internode we can do a bit better. Currently the way we send our data is through the CPU. 
NVIDIA GPUs now have an option called [InfiniBand GPUDirect Async](https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/)(IBGDA) which skips the CPU layer and enables our SMs to put the data directly onto the NIC. We can enable this
simply by setting `NVSHMEM_IB_ENABLE_IBGDA=1`. With it our internode speed jumps to 96 GB/s!

This works surprisingly well but what we truly want is to be able to operate on the received data without exiting our kernel, this leads us to

### Signaling

For that use cases NVSHMEM provides us with a set of signaling operations, they allow us to send notifications to the peer.
The two functions that we are the most interested in are:

`nvshmemx_signal_op` 

This one takes a pointer to the signal living on the symmetric heap, the value for the signal and the operation.

and `nvshmem_signal_wait_until` which takes the signal pointer, a predicate we want to run and the expected value.

This is how the updated code looks like

```C
template <typename scalar_t>
__global__ void exchange(scalar_t* destination, scalar_t* buffer, uint64_t* signal, int peer, int packet_size) 
{
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint64_t block_size = blockDim.x * packet_size;

    nvshmemx_putmem_block(destination + off, buffer + off, block_size, peer);
    nvshmem_fence();
    __syncthreads();

    constexpr uint64_t SIG_SYNC = 1;
    if (threadIdx.x == 0)
    {
        nvshmemx_signal_op(signal + blockIdx.x, SIG_SYNC, NVSHMEM_SIGNAL_SET, peer);
    }
    nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_EQ, SIG_SYNC);

    for (int i = threadIdx.x; i < block_size/(sizeof(scalar_t)); i += blockDim.x)
        buffer[off+i] = destination[off+i];
}

void exchange(torch::Tensor& buffer, int packet_size, int block_size, int peer) 
{
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    half *destination = (half *) nvshmem_malloc(buffer.numel() * sizeof(half));

    nvshmemx_buffer_register(buffer.data_ptr(), buffer.numel() * sizeof(half));
    
    const uint32_t grid_size = std::ceil(buffer.numel()*sizeof(half) / float(packet_size*block_size));

    uint64_t *signal = (uint64_t *) nvshmem_malloc(grid_size * sizeof(uint64_t));

    exchange<<<grid_size, block_size, 0, stream>>>(destination,
            static_cast<half*>(buffer.data_ptr()),
            signal,
            peer,
            packet_size);

    nvshmemx_barrier_all_on_stream(stream);

    nvshmemx_buffer_unregister(buffer.data_ptr());
    nvshmem_free(destination);
}
```

Right now in our kernel, after placing the data on the remote PE we:
- Call `nvshmem_fence`, this ensures that all of the `put` operations will finish before we issue our signal
- Synchronize the threads
- Send the signal informing the other peer that we finished the operation
- All threads wait until they receive the signal
- Every thread copies data back to destination

Let's run our sweep again. With this we're getting 80 GB/s internode and 560 GB/s intranode

Okay, we're actually doing operations on the data inside our kernel but we got a big slowdown, can we go faster?

Of course we can.

First we can replace the `putmem_block` + `signal_op` with a fused `putmem_signal_block` function, this handles all of the synchronization for us and packs the data together:

```C
nvshmemx_putmem_signal_block(destination + off, buffer + off, block_size, signal + blockIdx.x, SIG_SYNC, NVSHMEM_SIGNAL_SET, peer);
```

Second, `signal_wait_until` is a function that reads from memory, it's a slow operation and we're better off doing it on one thread only and then 
synchronizing 
```C
if (threadIdx.x == 0)
    nvshmem_signal_wait_until(signal + blockIdx.x, NVSHMEM_CMP_EQ, SIG_SYNC);
__syncthreads();
```

Lastly we can process our data in vectorized form to increase bytes in flight:
```C
for (int i = threadIdx.x; i < block_size/(sizeof(float4)); i += blockDim.x)
    reinterpret_cast<float4*>(buffer + off)[i] = reinterpret_cast<float4*>(destination + off)[i];
```

Sweeping again we're back at 96 GB/s for internode and 733 GB/s for intranode!

Okay, this looks pretty nice to me, let's now jump to the juicy part

## AllReduce


The algorithm that I'll use for implementing all reduce is a ring algorithm. How it works is that all of the GPUs taking part in the operation
are connected in a ring-like fashion, Then, each iteration they send a chunk of data to the peer in the next position, and receive a chunk of data
from the peer in the previous position. This way after `n_pes - 1` hops we get a full chunk of data. The algorithm operates in two phases:

Reduce phase where we pass the data that we received + our local data 

![ring_reduce](https://github.com/user-attachments/assets/2fddb359-9af7-48f7-b87c-4efa822df2aa)

Broadcast phase, where we propagate the final output through the ring 

![ring_broadcast](https://github.com/user-attachments/assets/473ad547-0624-4743-b161-171383e5423d)


### Coding our ring

The ring code looks like this:
- Determine what part of the data we send/recieve and to/from which peer
- Go into reduce loop 
- Send data to next peer in a ring
- Wait to receive data from the previous peer
- Perform a reduction
- Go into broadcast phase where we do the same thing, but this time saving the final output in our buffer

Communication wise it's all the same except that we now use `NVSHMEM_SIGNAL_ADD` to increment the stage in our buffer, and we compare
using `GE`, this is because the previous peer sends us more than one chunk, and if both arrive before we can process them we'll deadlock

```C
template <typename scalar_t>
__global__ void all_reduce_ring_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
    const uint32_t block_size = blockDim.x * packet_size;
    const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    int send_peer = (pe+1) % n_pes;
    int recv_peer = (n_pes + pe-1) % n_pes;
    int ring_pos = pe;

    int send_chunk = ring_pos % n_pes;
    int recv_chunk = (n_pes + ring_pos-1) % n_pes;

    uint64_t* local_signal = signal + blockIdx.x;
    // REDUCE PHASE
    for (int chunk = 0; chunk < n_pes - 1; chunk++)
    {
        nvshmemx_putmem_signal_nbi_block(destination + off + send_chunk*chunk_off,
                buffer + send_chunk*chunk_off + off,
                block_size, local_signal, 1, NVSHMEM_SIGNAL_ADD, send_peer);

        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);
        __syncthreads();

        for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
        {
            buffer[recv_chunk*chunk_off + off + i] += destination[off+ recv_chunk*chunk_off + i];
        }

        stage++;
        send_chunk = recv_chunk;
        recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
    }

    // BROADCAST PHASE
    for (int chunk = 0; chunk < n_pes - 1; chunk++) 
    {
        nvshmemx_putmem_signal_nbi_block(destination + off + send_chunk*chunk_off,
                buffer + send_chunk*chunk_off + off,
                block_size, local_signal, 1, NVSHMEM_SIGNAL_ADD, send_peer);

        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_GE, stage);
        __syncthreads();

        for (int i = threadIdx.x; i < block_size/sizeof(scalar_t); i += blockDim.x)
        {
            buffer[recv_chunk*chunk_off + off + i] = destination[off + recv_chunk*chunk_off + i];
        }
        stage++;
        send_chunk = recv_chunk;
        recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
    }
}
```

Let's check the bandwidth that we're getting with this kernel(sweeping across `block_size` and `packet_size`)
<img alt="comparison_6_nccl_intra_vs_penny_intra" src="https://github.com/user-attachments/assets/316b2367-d521-4a74-b032-2fac5f42edae" />

Okay, it's not that bad but let's see how can we improve on this.

First let's start with loading our value in 16B chunks, since we're gonna go through the data later float4 is a pain to use but we can do this by using a struct that promises alignment:
```C
template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};
```
We can then change our reduction/broadcasting loops to this:


```C
using P = array_t<scalar_t, 16/sizeof(scalar_t)>;
```
```C
for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
{
    P buf = reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i];
    P dst = reinterpret_cast<P*>(destination + off + recv_chunk*chunk_off)[i];
    P res;
    for (int j = 0; j < P::size; j++)
        res.data[j] = float(buf.data[j]) + float(dst.data[j]);
    reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i] = res;
}
```
```C
for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
{
    reinterpret_cast<P*>(buffer + recv_chunk*chunk_off + off)[i] =
        reinterpret_cast<P*>(destination + off + recv_chunk*chunk_off)[i];
}
```
The next thing we can fix is that we used `ADD` to change our signal, it turns out that it's not the optimal way to do this because it does an atomic operation
and we know that only this peer will change this value and we know what value we want to change it to. Let's switch to 
```C
nvshmemx_putmem_signal_nbi_block(destination + off + chunk*chunk_off,
        buffer + send_chunk*chunk_off + off,
        block_size, local_signal, stage, NVSHMEM_SIGNAL_SET, send_peer);
```
<img alt="comparison_7_nccl_intra_vs_penny_intra_opt" src="https://github.com/user-attachments/assets/2733c75c-f3f3-40ee-97e9-3de261b99f5d" />

Okay now the speeds we are getting are *almost* satisfying. The one thing that stands out is how bad we are compared to NCCL on small buffers.
What I've noticed is that up to a certain points, they are all the same speed, this means that we're very heavily latency bound. This lead me to
write a new kernel for those:

### Simple ring kernel

Because our small buffer sends are very latency bound, we essentially want to reduce the amounts of hops in our ring and increase the amount of data
that we're sending. Using chunks to send the data was bandwidth efficient but it gave us a constraint on how big of a packet can we send through the network,
the maximum was `packet_size * block_size * world_size`, we can drop the `world_size` scale by doing a simple ring that doesn't deal in chunks

At the cost of worse parallelism we're getting less hops per peer. Previously each was sending `2*(world_size-1)` small chunks, now each is sending `2` big chunks

![Simple_ring](https://github.com/user-attachments/assets/5061cd4c-632d-4e09-bf67-b5830af2e8a2)

The code has this structure:

- Initialize our ring variables
- Start by sending the data from position 0
- All other ranks wait for the data do the reduction and propagate it down the ring
- Last node leaves because it has the final output
- All other nodes broadcast the data down the ring

```C
template <typename scalar_t>
__global__ void all_reduce_simple_ring_kernel(scalar_t* __restrict__ destination, scalar_t* __restrict__ buffer, uint64_t* __restrict__ signal,
        const int packet_size, const int gpus_per_node, int stage)
{
    using P = array_t<scalar_t, 16/sizeof(scalar_t)>;

    const uint32_t block_size = blockDim.x * packet_size;
    const uint64_t off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);

    const int pe = nvshmem_my_pe();
    const int n_pes = nvshmem_n_pes();

    int send_peer = (pe+1) % n_pes;
    int recv_peer = (n_pes + pe-1) % n_pes;
    int ring_pos = pe;

    int send_chunk = ring_pos % n_pes;
    int recv_chunk = (n_pes + ring_pos-1) % n_pes;

    uint64_t* local_signal = signal + blockIdx.x;
    int send_stage = stage;
    int recv_stage = stage;

    // ring 0 initializes the send
    if (ring_pos == 0)
    {
        nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off),
                reinterpret_cast<float4*>(buffer + off),
                block_size, local_signal, send_stage, NVSHMEM_SIGNAL_SET, send_peer);
        send_stage++;
    }
    else 
    {
        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_EQ, recv_stage);
        __syncthreads();
        recv_stage++;

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            P buf = reinterpret_cast<P*>(buffer + off)[i];
            P dst = reinterpret_cast<P*>(destination + off)[i];
            P res;
            for (int j = 0; j < P::size; j++)
                res.data[j] = float(buf.data[j]) + float(dst.data[j]);
            reinterpret_cast<P*>(buffer + off)[i] = res;
        }
        nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off),
                reinterpret_cast<float4*>(buffer + off),
                block_size, local_signal, send_stage, NVSHMEM_SIGNAL_SET, send_peer);
        send_stage++;
    }

    // last node has the final position, it does not need to wait or store
    if (ring_pos != n_pes - 1)
    {
        if (threadIdx.x == 0)
            nvshmem_signal_wait_until(local_signal, NVSHMEM_CMP_EQ, recv_stage);
        __syncthreads();

       // last node has the final position, we don't need to send it the data again
       if (ring_pos < n_pes - 2)
            nvshmemx_putmem_signal_nbi_block(reinterpret_cast<float4*>(destination + off),
                    reinterpret_cast<float4*>(destination + off),
                    block_size, local_signal, send_stage, NVSHMEM_SIGNAL_SET, send_peer);

        for (int i = threadIdx.x; i < block_size/(sizeof(P)); i += blockDim.x)
        {
            reinterpret_cast<P*>(buffer + off)[i] =
                reinterpret_cast<P*>(destination + off)[i];
        }
    }
}
```

Let's benchmark it
<img alt="comparison_8_nccl_intra_vs_penny_intra_combined" src="https://github.com/user-attachments/assets/737a6a02-661f-4933-9a49-b4c18e9e9cfb" />

It's much better but still behind NCCL, for this part of the blogpost I'll say I'm satisfied with it but in reality I'm not. We'll get back to fixing this later. For now let's jump into

## Multi node reduction

So we've kinda cracked single node, let's run our kernel in a multi node setting 
<img alt="comparison_1_nccl_ring_vs_penny_base" src="https://github.com/user-attachments/assets/7dbe944d-01e7-42bd-b323-aeea6f4ba184" />

Wow it's quite bad.

To understand why this happens we need to visualize our ring. If you remembered from the introduction on communications. We send and receive data internode through our NICs
Currently our ring only utilizes two of them on each node for communication.
<img alt="INTRANODE_RING" src="https://github.com/user-attachments/assets/8542334e-9231-4cdc-89d7-ee6e3e906206" />

If we rerun our ring reduction kernel with a tool that can analyze traffic like [ibtop](https://github.com/JannikSt/ibtop) we can confirm that only one of our NICs is sending the data and only one is receiving the data:
<img alt="ibtop_output" src="https://github.com/user-attachments/assets/df298fd3-2228-4344-ae76-0c4955c089f7" />

### Solution

The solution to this problem is to run as many rings as we have NICs, each ring will send data through one NIC and receive data through a second one

The very important part is how we can structure our rings. For this we would need to understand the higher level of communication. Our NICs inside the node are connected to a number of leaf switches.
Which are in turn connected to a number of spine switches. There are a lot of configurations of how leaf switches are connected but for AI workloads the typical solution would be a rail optimized design.
In this way, same index NICs on nodes are connected to the same leaf switch, so if we were to create a ring on nodes being on the same leaf, it would be possible to do so with just one hop, without ever hitting the spline switch.

<img alt="SWITCHES" src="https://github.com/user-attachments/assets/1301f39e-fd3a-43d6-9a1c-58637e19525c" />

This is the idea behind alternating rings. It's designed for rail-optimized topologies and it ensures that we don't cross the rails between NICs. Here every other node alternates the ring so that we can send data internode through NICs with the same index 
<img alt="RING1" src="https://github.com/user-attachments/assets/483bd1d9-cc87-47e4-9e53-d026e492ab2f" />


We can create this type of ring for every pair of NICs
<img alt="RING2" src="https://github.com/user-attachments/assets/94375e64-430d-4ad8-b867-e8cd2960d63d" />


Since the bandwidth is bidirectional, we can invert every other ring for that extra speed improvement


In code the initialization now looks like this:
```C
const uint64_t base_off = (blockIdx.x * blockDim.x) * packet_size/sizeof(scalar_t);
const uint32_t block_size = blockDim.x * packet_size;
const uint64_t chunk_off = (gridDim.x * blockDim.x) * packet_size/sizeof(scalar_t);
const uint32_t ring_id = blockIdx.y;
const uint64_t ring_off = ring_id * chunk_off * nvshmem_n_pes();
const uint64_t off = base_off + ring_off;

const int pe = nvshmem_my_pe();
const int n_pes = nvshmem_n_pes();


int send_peer;
int recv_peer;
int ring_pos;

if constexpr (INTERNODE)
{
// TODO this is currently a hack to get the ring position, since it changes a lot
// it's easier to find it than to derive an expression for it
    int curr_pe = -1;
    send_peer = 0;
    ring_pos = -1;
    while (curr_pe != pe)
    {
        curr_pe = send_peer;
        int curr_node = curr_pe/gpus_per_node;
        int curr_rank = curr_pe%gpus_per_node;
        // Send PE on even nodes, Recv PE on odd ones
        if (curr_rank == (ring_id/2)*2)
        {
            if (curr_node%2 == 1)
            {
                send_peer = curr_node * gpus_per_node + (gpus_per_node + curr_rank - 1) % gpus_per_node;
                recv_peer = (n_pes + curr_pe - gpus_per_node) % n_pes;
            }
            else
            {
                send_peer = (n_pes + curr_pe + gpus_per_node) % n_pes;
                recv_peer = curr_node * gpus_per_node + (gpus_per_node + curr_rank - 1) % gpus_per_node;
            }
        }
        // Recv PE on even nodes, Send PE on odd ones
        else if (curr_rank == (ring_id/2)*2 + 1)
        {
            if (curr_node%2 == 1)
            {
                send_peer = (n_pes + curr_pe + gpus_per_node) % n_pes;
                recv_peer = curr_node * gpus_per_node + (curr_rank + 1) % gpus_per_node;
            }
            else
            {
                send_peer = curr_node * gpus_per_node + (curr_rank + 1) % gpus_per_node;
                recv_peer = (n_pes + curr_pe - gpus_per_node) % n_pes;
            }
        }
        //intranode
        else
        {
            send_peer = curr_node*gpus_per_node + (curr_rank+1) % gpus_per_node;
            recv_peer = curr_node*gpus_per_node + (gpus_per_node + curr_rank-1) % gpus_per_node;
            // Odd nodes need to alternate direction
            if (curr_node%2 == 1)
                swap_cu(send_peer, recv_peer);
        }
        ring_pos++;
    }
    int send_chunk = ring_pos % n_pes;
    int recv_chunk = (n_pes + ring_pos-1) % n_pes;
    // alternate every odd ring
    if(ring_id%2 == 1 && INTERNODE)
    {
        swap_cu(send_chunk, recv_chunk);
        swap_cu(send_peer, recv_peer);
    }
}
```

I do agree that it's pretty non pragmatic, especially the while loop, but it made it much easier to change how we structure our ring, and due to time 
constraints I didn't go through finding the heuristics to eliminate it(+ it doesn't affect performance so there was no pressure to do so). I'll probably refactor 
this later once I settle on heuristics
 
The rest of the code looks more less the same except for that the alternating rings need to increment `recv_chunk`
```C
if(ring_id%2 == 1 && INTERNODE)
    recv_chunk = (n_pes + recv_chunk + 1)%n_pes;
else
    recv_chunk = (n_pes + recv_chunk - 1)%n_pes;
```
<img alt="comparison_2_nccl_ring_vs_penny_ring" src="https://github.com/user-attachments/assets/d19ce1cb-6570-4a9b-a160-c845840b34aa" />

Running this we can see that our performance is much better although it's still a bit off. To our rescue comes another environment variable `NVSHMEM_IBGDA_NUM_RC_PER_PE`, this exposes the number of Reliable Connections(RC) in our peer. RCs are a 
type of a Queue Pair(QP) (so a pair of send and receive queue) used for reliable communication. You can think of this as the equivalent of a socket in networking. By default this was set to 2 but we can increase the number. For me 32+ started giving much better results.
<img alt="comparison_3_nccl_ring_vs_penny_ring_qp" src="https://github.com/user-attachments/assets/4d094fd6-7f50-4d90-8eec-a2377739ac06" />

This is a pretty awesome result for large buffers, again we can combine this with our simple ring from earlier to get performance in latency sensitive situations.

<img alt="comparison_4_nccl_ring_vs_penny_combined" src="https://github.com/user-attachments/assets/bfe3a9c4-7aa3-4021-81c9-89341a51eae5" />

Before you start making conclusions about how we got to beat NCCL for the high buffers, this plot is a bit of a lie. We forced 'NCCL_ALGO=RING' to compare apples to apples since we're implementing a ring algorithm here. But by default
NCCL chooses a tree algorithm. If we compare against that it turns out that we still have room for improvement. I started playing around with it for a bit but sadly no longer have access to a multinode setup (wink wink if you do and want to support 
educational content)
<img alt="comparison_5_nccl_tree_vs_penny_ring_qp" src="https://github.com/user-attachments/assets/d947106f-fdd2-493e-b696-c7d7fd0f384c" />


## Conclusion and next steps

NVSHMEM is quite awesome and after getting an intuition on it it was fairly easy to get good performance out of this. For the next part I would love to get the lower latency kernels up and running. 
I'll play around with [IPC Buffers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#interprocess-communication) as this seems to be the way to achieve [very low latency](https://github.com/vllm-project/vllm/pull/2192). Right now I also have no idea on how to profile this in a sensible way. It was all mostly world model based 
performance tuning, and checking NIC utilization for multi node setup(would love some good resources if you heard of some).

## Shameless self promotion

I'm also posting on [X](https://x.com/SzymonOzog_) and I have a [YouTube](https://www.youtube.com/@szymonozog7862) channel where you can get some more GPU content. If you liked the article too much you can always [Buy Me A Coffe](https://buymeacoffee.com/simonoz)


## Resources

[RDMA Aware Networks Programming User Manual](https://docs.nvidia.com/networking/display/rdmaawareprogrammingv17)
[NVSHMEM Documentation](https://docs.nvidia.com/nvshmem/api/index.html)
[DeepEP codebase](https://github.com/deepseek-ai/deepep)
[GPU MODE talk by Jeff Hammond](https://youtu.be/zxGVvMN6WaM?si=etPiCfP7cXaeJX1m)
[Training Deep Learning Models at Scale: How NCCL Enables Best Performance on AI Data Center Networks](https://www.nvidia.com/en-us/on-demand/session/gtc24-s62129/)
[Demystifying NCCL](https://arxiv.org/pdf/2507.04786)

