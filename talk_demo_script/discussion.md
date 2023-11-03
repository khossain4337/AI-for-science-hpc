# Markdown preview `command+k+v`

## Talk at SC23
This is an informal document for discussion between Riccardo and Khalid to collect answer to basic technology questions and clarifications.

- `cuDNN`: cuDNN is a GPU optimized Nvidia library to work with frameworks like Tensorflow and PyTorch as a backend. This in-principle works only at a single GPU level i.e. runs a single mpi rank. To do more than that, we need distributed training and other applications/library.

- `NCCL`: Collective communications library. This Nvidia's version of mpi -- possibly more efficient compared to plain mpi. This is needed to run on multiple devices within a node or accross multiple nodes. 

- `Horovod`: Horovod is another application that is used to make distributed training easier to implement, can be used with both TF and PT. Horovod used `NCCL` perform `mpi-communication-type` operations like _ring-allreduce_ across multiple GPUs within a node, and `MPI` to coordinate across the nodes communications. In general, Horovod should installed (built) with `NCCL` and `MPI` to perform distributed computing across multiple nodes and multiple GPUs within each of these nodes.

- `DDP`: DDP is a PT native API that is used to implement distributed training.

## Riccardo's Q/A:
Here we try to answer the questions Riccardo asked as the backbone of the sequential description of the full process of performing a run for our representative applications.

- Table 2 and 4 results (single GPU and single node results)
    - Use CT as the representative app
    - Is there anything special about the run script for Nvidia, AMD, or PVC? (e.g., special        environment flags)
        - There is nothing special about running on different devices. Running on different devices is just about setting up different libraries correctly. It does not require any special environment flags except 1 case. We will discuss that below.
        - __Primary requirements__
            - For Nvidia devices, we will need `cuDNN` and `NCCL`. 
                - `cuDNN` is required for utilizing the GPUs. This is the GPU optimized deep learning library that is used as a backend for ML frameworks -- we have used PyTorch and TensorFlow. This facilitates the training using 1 device (1 GPU, 1 `MPI` rank)
                - `NCCL` is required for establishing inter-GPU communication. This will allow us to do distributed training within a node with multiple GPUs.
            - For AMD devices, the equivalents would be `hipDNN` and `RCCL`, which come through the `ROCm sofware stack`.
            - For Intel devices, the equivalents would be `oneDNN` and `oneCCL`, which are part of the `oneAPI programming model`.
        - __Distributed training requirements__
            - For all the devices that we discuss here requires `API`s for distributed training. For our applications, we used `Horovod` and `DDP` in conjunction with PyTorch and TensorFlow.
            - `DDP` is PyTorch's native distributed *trainer* which coordinates the communication across multiple devices. Standard PyTorch installations enables the use of `DDP` right out of the box -- all we have to do is initialize `DDP` with specific device type.
            - `Horovod` requires a little extra care in terms of installation and should be built with `MPI` and appropriate communication libraries (`NCCL`, `oneCCL`, `RCCL`).
        - __Framework level requirements__
            - We could run on Nvidia and AMD devices with standard PyTorch and TensorFlow installations. 
            - For Intel devices, we need to use `intel extension for TensorFlow -- itex` and `intel extension for PyTorch -- ipex`
    - Handling different precision is done at the framework level, except for the case of TF32 and FP32.
        - For Nvidia devices, we use `export NVIDIA_TF32_OVERRIDE=0` environment variable to run the calculation with FP32, as the default is TF32.
        - For Intel devices, we use `export ITEX_FP32_MATH_MODE=TF32` environment variable for TensorFlow and `export IPEX_FP32_MATH_MODE=TF32` for PyTorch to run the calculation in TF32 as the default is FP32.
        - For reduced precision calculation, For Intel BF16 is the default and for Nvidia FP16 is the default.

If yes, highlight what.
If not, we can skip this part.
Maybe the TF32 vs. FP32 flags are the only important ones
