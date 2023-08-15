In this repository we present the supplemental material for our manuscript
"Demonstration of Portable Performance of Scientific Machine Learning on High 
Performance Computing Systems" submitted to SC 2023.

Short Description
=================

In this manuscript, we present training performance metric (throughput, Images/sec)
for a suite of scientific AI/ML applications across three state of the art 
hardware Nvidia A100, AMD Mi250, and Intel Data Center GPU Max 1550. 

In this repository, we will provide the installation instructions (which might
come from specific application repositories), and version details of specific
applications that we have used to produce the performance data.

List of applications
====================
The following is the list of applications that we have studied:

[CosmicTagger](https://github.com/coreyjadams/CosmicTagger)

[FFN](https://github.com/tomuram/ffn/tree/unified_stable)

[Atlas-pointnet](https://github.com/jtchilders/atlas-pointnet/tree/master)

Broad installation strategy
===========================
For most of our applications, we followed the strategy of ```pip``` installing the application dependencies on a ```conda``` environment. As we were looking at the GPU performances, we installed the GPU libraries (for example ```CUDA```, ```CUDA-toolkit```) through ```conda```, and ```pip``` install the frameworks (```PyTorch``` and ```TensorFlow```). The distributed training frameworks like ```Horovod```, was installed through ```pip```. In the  `versions.pdf` file at the top level of the repository has detailed information about the relevant packages.

### On machines available at ALCF
In practise, we have taken the advantage of pre-built modules on ALCF machines like `Polaris`, `JLSE nodes` and `Sunspot`. In each of our machines, we make a virtual environmnet with `--system-site-packages` from the base `conda` module, load appropriate compiler modules and `mpi` backends, and `pip` install frameworks with other dependencies in the virtual environment.

Running the applications
========================
- We have provided sample run scripts for our applications. These scripts display most of the relevant argument flags that are available from the application. 
- To get the results, we did our runs with `mpiexec` even for `rank 1` trainings and used appropriate cpu bindings to improve the single rank throughput.
- Specific environmental variables:
    - To run an application with `FP32` on `A100`, one must override the default `TF32` data type. The flag to do so
        - ```export NVIDIA_TF32_OVERRIDE=0```
    - To run an application with `TF32` on Intel Max GPU, one must set the following flag:
        - `export ITEX_FP32_MATH_MODE=TF32` (for TensorFlow)
        - `export IPEX_FP32_MATH_MODE=TF32` (for PyTorch)
        - These are Intel's TensorFlow and PyTorch extensions.

