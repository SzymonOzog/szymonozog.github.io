--- 
title: Writing my own communications library - a worklog of creating Penny Part 3 
date: 2025-11-11
---

This is a third part of my worklog documenting the creation of my own communications library, if you didn't read part one and two you might want to catch up 
[here](https://szymonozog.github.io/posts/2025-09-21-Penny-worklog-1.html) and [here](https://szymonozog.github.io/posts/2025-10-26-Penny-worklog-2.html). If you already have read them or you like to start from the middle please continue.

# Previously on Penny

What we did last time was we took a look at the one-shot and two-shot all reduce algorithms that are implemented in vLLM for low latency communications. We reproduced the results on a single node and caught up to or outperformed NCCL
on all buffer sizes. For the smallest buffers we're slightly behind on vLLM but there is a good excuse for it(read part 2).

<img alt="cumulative2" src="https://github.com/user-attachments/assets/834d17ba-e355-443b-b13b-d2654786bc0d" />

# Return of the multinode

Now that we've got single node correct it's time to close the multinode gap. If you recall last time we left off with being on par with NCCL for the ring allreduce for medium and larger sized buffers.

My goal for this part is to get a low latency allreduce working multinode. I'm not aware of anyone providing working open source code for it so Penny might be the first open source project officially supporting this(spoiler alert: it works)

The starting point will be vLLM's custom allreduce(once again all credit for the base code to Hanzhi Zhou who introduced it in this [PR](https://github.com/vllm-project/vllm/pull/2192). I won't go over most of the code as it's 
very similar to our results from the previous parts and because the author of the PR explains it very well(also the code is quite easy to read once you get the algorithm behind it)

Algorithmically they're almost the same as our solution from the previous part of the worklog but instead of using NVSHMEM they use IPC buffers, that have the limitations of only working intranode. However, with NVSHMEM
we have all the power in the world to make the algorithms work internode.

# The grand strategy

The plan for how to do it is simple:

- Perform intranode reduction using IPC buffers
- Now all GPUs on the node contain the same full reduction
- Exchange the buffer between nodes
- Add to current result

For oneshot it would look like this:
<img alt="OneshotIntranode" src="https://github.com/user-attachments/assets/912478fe-dafb-491a-ac7a-637b93d32a67" />

For twoshot we need to perform the gather phase afterwards, but this can also be done with IPC buffers reusing
the existing codepath if we perform the initial reduction with bigger granularity. 

<img alt="TwoshotIntranode" src="https://github.com/user-attachments/assets/3d41783a-7396-405e-8ff0-8c6fd5d85f61" />

# Going through changes

First let's take a look at the oneshot allreduce

There are a few major changes that we need to make, I've commented around all of them

First, if we're running this intranode instead of saving the intermediate results to the output buffer, we store them on our local part of the symmetric memory buffer

Second change is how we update the flag. Custom all reduce holds a counter on how much time each block has synchronized, however for internode communication, we don't want each block 
to send a small chunk of the data. We want it all and we want it ~~now~~ as soon as the results are ready. There is an issue though, if we first run a small bach size that launches 1 block
and then a big batch size that launches more blocks, the signal variable on each block will hold a different value. Hence we update all signals to the same value on `barrier_at_end`


```C
template <typename T, int ngpus, int nnodes = 1>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_1stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size, 
                               void* buffer, uint64_t * signal) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  // note: we don't reorder the address so the accumulation order is the same
  // for all ranks, ensuring bitwise identical results
  auto dp = *_dp;

  //CHANGE get node ID, if we're internode, perform the reduction to symmetric memory buffer
  const int pe = nvshmem_my_pe();
  const int node = pe/ngpus;
  P* local_buffer = nnodes == 1 ? reinterpret_cast<P*>(result) 
                                : reinterpret_cast<P*>(buffer) + node*size;
  barrier_at_start<ngpus>(sg, self_sg, rank);
  // do the actual reduction
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    local_buffer[idx] = packed_reduce<P, ngpus, A>((const P**)&dp.ptrs[0], idx);
  }

 //CHANGE, barrier_at_end for multinode needs to update all flags(see code block below)
  barrier_at_end<ngpus, true, (nnodes>1)>(sg, self_sg, rank);
  if (nnodes == 1)
      return;

 //CHANGE, exchange reduction results across nodes 
  __syncthreads();
  uint32_t new_signal = self_sg->_flag[blockIdx.x] + 1;
  uint64_t* local_signal = signal;
  if (blockIdx.x < nnodes && blockIdx.x != node)
  {
      int send_node = blockIdx.x;
      int exchange_pe = (rank + send_node*ngpus)%(nnodes*ngpus);
      nvshmemx_putmem_signal_nbi_block(reinterpret_cast<P*>(buffer) + node*size,
              local_buffer, size*sizeof(P),
              local_signal + node, new_signal, NVSHMEM_SIGNAL_SET, exchange_pe);
  }

  if(threadIdx.x < nnodes && threadIdx.x != node)
  {
      nvshmem_signal_wait_until(local_signal + threadIdx.x, NVSHMEM_CMP_EQ, new_signal);
  }
  __syncthreads();
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
        P res = local_buffer[idx];
        for (int recv_node = 0; recv_node<nnodes; recv_node++)
        {
            if(recv_node == node)
                continue;
            P buf = reinterpret_cast<P*>(buffer)[idx+recv_node*size];
            for (int j = 0; j < P::size; j++)
            {
                res.data[j] += float(buf.data[j]);
            }
        }
        reinterpret_cast<P*>(result)[idx] = res;
  }
```
As noted before we also need to change the barrier code to account for different grid sizes

```C
template <int ngpus, bool final_sync = false, bool internode = false>
DINLINE void barrier_at_end(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
  uint32_t flag = self_sg->_flag[blockIdx.x] + 1;
  if (threadIdx.x < ngpus) {
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->end[blockIdx.x][rank];
    auto self_counter_ptr = &self_sg->end[blockIdx.x][threadIdx.x];
    // Write the expected counter value to peer and wait for correct value from
    // peer.
    if constexpr (!final_sync) {
      st_flag_release(peer_counter_ptr, flag);
      while (ld_flag_acquire(self_counter_ptr) != flag);
    } else {
      st_flag_volatile(peer_counter_ptr, flag);
      while (ld_flag_volatile(self_counter_ptr) != flag);
    }
  }
  if constexpr (!final_sync) __syncthreads();

  // use one thread to update flag
  if (threadIdx.x == 0) self_sg->_flag[blockIdx.x] = flag;
  
  // CHANGE, make sure all flags are updated
  if constexpr(internode)
  {
      if (blockIdx.x == 0 && threadIdx.x >= gridDim.x && threadIdx.x < kMaxBlocks)
          self_sg->_flag[threadIdx.x] = flag;
  }
}
```

For 2 stage the flow is very similar, except this time after the internode exchange we need to propagate the results across nodes

```C
template <typename T, int ngpus, int nnodes = 1>
__global__ void __launch_bounds__(512, 1)
    cross_device_reduce_2stage(RankData* _dp, RankSignals sg, Signal* self_sg,
                               T* __restrict__ result, int rank, int size, 
                               void* buffer, uint64_t * signal) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  int part = size / ngpus;
  int start = rank * part;
  int end = rank == ngpus - 1 ? size : start + part;
  int largest_part = part + size % ngpus;
  const P* ptrs[ngpus];
  P* tmps[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    int target = (rank + i) % ngpus;
    ptrs[i] = (const P*)_dp->ptrs[target];
    tmps[i] = get_tmp_buf<P>(sg.signals[target]);
  }
  auto tmp_out = tmps[0];

  const int pe = nvshmem_my_pe();
  const int node = pe/ngpus;
  // CHANGE, save to symmetric memory if internode
  P* local_buffer = nnodes == 1 ? tmp_out : reinterpret_cast<P*>(buffer) + node*part;
  barrier_at_start<ngpus>(sg, self_sg, rank);

  // stage 1: reduce scatter
  for (int idx = start + tid; idx < end; idx += stride) {
    local_buffer[idx - start] = packed_reduce<P, ngpus, A>(ptrs, idx);
  }
  barrier_at_end<ngpus, false, (nnodes>1)>(sg, self_sg, rank);

 //CHANGE, exchange reduction results across nodes 
  if (nnodes > 1)
  {
      uint64_t* local_signal = signal;
      uint32_t new_signal = self_sg->_flag[blockIdx.x] + 1;
      const int pe = nvshmem_my_pe();

      if (blockIdx.x < nnodes && blockIdx.x != node)
      {
          int send_node = blockIdx.x;
          int exchange_pe = (rank + send_node*ngpus)%(nnodes*ngpus);
          nvshmemx_putmem_signal_nbi_block(reinterpret_cast<P*>(buffer) + node*part,
                  local_buffer, part*sizeof(P),
                  local_signal + node, new_signal, NVSHMEM_SIGNAL_SET, exchange_pe);
      }

      if(threadIdx.x < nnodes && threadIdx.x != node)
      {
          nvshmem_signal_wait_until(local_signal + threadIdx.x, NVSHMEM_CMP_EQ, new_signal);
      }
      __syncthreads();
      for (int idx = start + tid; idx < end; idx += stride) {
          P res = local_buffer[idx - start];
          for (int recv_node = 0; recv_node<nnodes; recv_node++)
          {
              if(recv_node == node)
                  continue;
              P buf = reinterpret_cast<P*>(buffer)[idx-start + recv_node*part];
              for (int j = 0; j < P::size; j++)
              {
                  res.data[j] += float(buf.data[j]);
              }
          }
          reinterpret_cast<P*>(tmp_out)[idx-start] = res;
      }
      barrier_at_end<ngpus, false, (nnodes>1)>(sg, self_sg, rank);
  }

  // stage 2: allgather. Note: it's important to match the tid between
  // the two stages, because visibility across devices is only guaranteed
  // between threads that have the same tid. If thread i computes the sum of
  // start + i in the first stage, then thread i also gathers start + i from
  // all ranks.

  for (int idx = tid; idx < largest_part; idx += stride) {
#pragma unroll
    for (int i = 0; i < ngpus; i++) {
      int gather_from_rank = ((rank + i) % ngpus);
      if (gather_from_rank == ngpus - 1 || idx < part) {
        int dst_idx = gather_from_rank * part + idx;
        ((P*)result)[dst_idx] = tmps[i][idx];
      }
    }
  }
}
```

There were also a few more boring changes for which I'm not putting the code in here. Mostly allocating the symmetric memory buffer and signal, 
ensuring we save our own intranode buffer handles and that we correctly send cuda graph registered buffers


# Are we faster?

Now that the code is ready it's time to benchmark. I rented up to 4 nodes and ran the tests, the results are quite solid. We manage to outperform NCCL on buffers up to 8MB which covers most LLM serving scenarios.

<img alt="combined" src="https://github.com/user-attachments/assets/cb1bbf75-bc64-44ac-90c8-aaca98ce2433" />

# How do we use it again? 

The usage is actually pretty simple, for sglang it's a matter of opening 

`python/sglang/srt/distributed/parallel_state.py`

and changing: 
```py
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
        )
```
to
```py
from penny.custom_all_reduce import CustomAllreduce

```
For detailed installation instructions of installing Penny please follow the [readme](https://github.com/SzymonOzog/Penny/blob/main/README.md)


# Are we working

Time to check for correctness, for this I decided to follow SGLang's [docs](https://docs.sglang.ai/developer_guide/contribution_guide.html#test-the-accuracy) and running gsm8k. The model I'm using is Qwen3 235B TP16 across 2 nodes

The results:
```
// NCCL
Accuracy: 0.900
Invalid: 0.000
Latency: 34.900 s
```

```
// Penny 
Accuracy: 0.915
Invalid: 0.000
Latency: 21.449 s
```

Okay, no accuracy drops. This is a good sign, now it's time for 

# E2E benchmarks

For this once again I followed the [docs](https://docs.sglang.ai/developer_guide/benchmark_and_profiling.html) of sglang and ran their serving benchmarks with the same setup as the corectness check

<img alt="image" src="https://github.com/user-attachments/assets/90add76c-a78d-4199-b95d-3a12736db6d5" />

First there is an excitement phase seeing the numbers go up. Then, I actually did the napkin math. If the kernel is 10-80% faster a 50% throughput increase seems highly unlikely.


# How are we so fast

Obviously I had to investigate what on earth had happened here. I relaunched both configurations and took out the traces 

When running with NCCL there seems to be a gap between consecutive batches.

<img alt="image" src="https://github.com/user-attachments/assets/d3c9da50-11f3-40b1-94dc-53a04731a5bb" />

The same gap is not present when running with Penny.

<img alt="image" src="https://github.com/user-attachments/assets/3f892bef-ebbd-46bb-a9d1-0c44afbfde13" />

I've looked more closely into it and it seems like with NCCL cudaGraphLaunch becomes a blocking operation, it doesn't return execution back to the CPU hence the worker/scheduler overlap in SGLang is disabled.

<img alt="Graph Issue" src="https://github.com/user-attachments/assets/8a7ed7ea-c643-4894-b28c-37581bf7da91" />

I did run into this bug previously and while it was tempting to say that I improved throughput by 50% I do want a fair comparison.

I'm not quite sure what causes this and I would love to investigate but running multiple nodes aint cheap so I decided to just roll back to an older version of SGLang(0.4.6-post5) where I remembered 
this was not occurring and reran the benchmarks to get those results:

<img alt="image" src="https://github.com/user-attachments/assets/67136f17-4270-49db-a1e7-51d6f54c01f9" />

26.3% average throughput increase! Pretty neat, before we start celebrating let's check if our graph caching issue is truly gone

<img alt="graph_fix" src="https://github.com/user-attachments/assets/6a362c8f-4ce1-4dea-916f-5a607e5b7949" />

Both graphs are launching non blocking, no cheating here, let's look at the flamegraph of both forward passes to check if the speed up is indeed due to our allreduce:

<img alt="Flamegraph" src="https://github.com/user-attachments/assets/422ec88d-d90a-4b1e-a0b0-5443228ea27d" />

All of the speedup does come from the allreduce that is 2x faster now. Let's verify if this makes sense. 

This is running a batch size of 32 with a hidden dimension of 4096 and a dtype 
of bfloat16 sending 32*4096*2 = 262144B of data. Looking at the speedup in an isolated scenario that translates to a 150% faster not 200%! 

I've pondered it for a while and I think the reason for this is our benchmark being performed in perfect conditions. 
All of the nodes run a big kernel and synchronize afterwards so that the CPU and synchronization overheads are gone. In a real world scenario, before
running allreduce all of the nodes need to arrive on the kernel. In our case it's slightly relaxed because we don't need to synchronize all of the PEs at all times. We have the initial synchronization step
where a node synchronizes intranode and performs the first reduction step that is sent to symmetric memory on other nodes. Only then we wait for other nodes to arrive. This way we achieve a slightly 
better parallelization of reduction and synchronization. Note however that this is just my theory that I'm unable to confirm(If you're interested in profiling this and giving your insights, Penny is a working group
on [GPU Mode](https://www.gpumode.com/v2/home)

# Conclusion

If you remembered correctly in part 1 I've stated `A goal of mine would be to be able to swap Penny and NCCL in an LLM serving framework and see close to no performance degradation.` 
Today we managed to achieve not only this but to actually improve the speed when running internode. I didn't expect this outcome but I'm very proud of it. It was an incredible journey 
and I've learned a lot about GPU communication. 

And yet I still have a few ideas I would like to implement.

See you in part 4




