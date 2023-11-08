# Script for SC23 talk
This document is the script for the talk. We are writing down the words that will be delivered during the presentation.

## Slide 1
Hi everyone, I am Khalid, and today we will talk about the performance portability of Python AI 
applications in HPC systems. We will also give a very short demo of portability across a couple of 
devices and present a few important results that my collaborators and I have produced during the 
course of our study.

## Motivation Slide
I sincerely believe that this audience does not need any convincing about the usefulness of running 
Python at scale, specially at this point of time when Python is the go to language for the AI 
community. But it is important to know how portable these Python applications can be __out of the 
box__, what level of performance we might expect with minimal change to the source code and what could 
be a possible way to improve upon what we have already.

We have chosen our applications from diverse scientific disciplines. These apps use various popular ML kernels like convolutions and pooling.

Hopefully, this presentation will convince you of at least two things: first, applications developed 
within frameworks like PyTorch and Tensorflow can deliver portable performance across many of the 
state-of-the art hardware, with minimal change to the source code. Second, for smaller models, which 
do not saturate both the memory and compute of a tile or a GPU, running multiple instances on a tile 
or GPU may lead to better performance in terms of overall throughput. 

Now we will look at the hardware that we have used.

## Hardware Slide
We have run our calculation on ALCF machines Polaris, JLSE AMD nodes, and Sunspot. Each Polaris node 
has 4 Nvidia A100 GPUs. For an AMD node, we have 4 MI250 GPUs, total 8 GCDs. A Sunspot node has 6 
Intel GPUs, in total of 12 tiles. When we say `Single GPU` we mean 1 A100, 2 GCDs of an MI250 and 2 
tiles of an Intel Max GPU.

Now we will introduce the applications that we have used for this study. We have collected data for several applications, but in the interest of time we will talk about the two applications we have used for demo.

## CT Slide
CosmicTagger, CT, comes from a neutrino physics research and perform a pixel level segmentation task 
to isolate the neutrino events using a modified UNet architecture. We implemented our schemes using 
both PyTorch and Tensorflow, and used Horovod for distributed training with TensorFlow, and DDP for 
PyTorch. The application uses 2D convolution with batch normalization and a LeakyReLU activation 
function.

## FFN Slide
The second application is the Flood Filling Network, FFN. Also performs a segmentation task, for mouse 
brain images to separate blood vessels, brain tissues etc. to build a complete catalogue of the brain. 
The application uses 3D convolutions with ReLU activations with skipp connections. The application is 
built using TensorFlow, and distributed training done using Horovod.

## AnisoSGS Slide
The next application comes from fluid dynamics. AnisoSGS, Anisotropic Sub-grid Stress model, attempts 
to construct a data-driven sub-grid stress model for filtered Navier-Stokes equation. The application uses Multi-layer perceptrons with LeakyReLU activation. The implementation is done using PyTorch and 
distributed training is performed through DDP.

## QCNN Slide
Quadrature Convolutional Neural Network is an application that develops an autoencoder to compress the 
solution states obtained from CFD simulations of unsteady flows on unstructured grids. This 
application uses MLP with batch normalization and max pooling followed by a Gaussian Linear Unit 
activation. The scheme is implemented using PyTorch and the distributed training is done using DDP.

## PointNet Atlas Slide
PointNet Atlas performs a pointcloud segmentation on detector data from particle collision simulations 
to isolate electron jets. The model uses MLPs and pooling layers with a ReLU activation.

## Demo Slide:
Now, we will show a demo of running CT on Sunspot, ALCF machine, with Intel GPU. We would love to 
demonstrate on another machine, Polaris with Nvidia GPUs, but for the sake of time we are restricting 
ourselves to one.

## Demo Narrative:
Here is a script that I will use to run on Sunspot from an interactive session. We can use similar 
script for a full batch submission mode. In that case we have to fill in the PBS directives in the 
beginning of the script. 

First we determine the work directory. The work directory is where the repository is cloned, and we 
will running a script called `bin/exec.py` which calls the trainer script. 

To run the script we need have dependencies installed correctly, the details of the dependencies can 
be found in the repository linked in the paper. The application itself is publicly available. In our 
case, we will use a pre-installed module. And we do that by using a module load command.

At the beginning, 

Then, to use the GPUs, we import `intel_exptension_for_pytorch as ipex` and use that to set the context for the device.`ipex` does that for the Intel GPU. We have to choose `compute_mode=XPU` and set the default device correctly. Which 
is done through this way: (Lines 600 - 634)
```
    def default_device_context(self):


        if self.args.run.compute_mode == ComputeMode.GPU:
            return torch.cuda.device(0)
        elif self.args.run.compute_mode == ComputeMode.XPU:
            # return contextlib.nullcontext
            try:
                return ipex.xpu.device("xpu:0")
            except:
                pass
            try:
                return ipex.device("xpu:0")
            except:
                pass
            return contextlib.nullcontext()
        # elif self.args.run.compute_mode == "DPCPP":
        #     return contextlib.nullcontext()
        #     # device = torch.device("dpcpp")
        else:
            return contextlib.nullcontext()
            # device = torch.device('cpu')

```
```
    def default_device(self):

        if self.args.run.compute_mode == ComputeMode.GPU:
            return torch.device("cuda")
        elif self.args.run.compute_mode == ComputeMode.XPU:
            device = torch.device("xpu")
        # elif self.args.run.compute_mode == "DPCPP":
        #     device = torch.device("dpcpp")
        else:
            device = torch.device('cpu')
        return device

```
Device `XPU` is specific to Intel GPUs.

Let's say, we want to do the training using FP32, in that case I choose the command line argument for 
`float32`. The application also supports BF16 for reduced precision (default on an Intel GPU) and TF32 
also. This choice takes effect through the following code section:

```
            if self.args.data.synthetic:
                minibatch_data['image'] = minibatch_data['image'].float()
                minibatch_data['label'] = minibatch_data['label']

            if self.args.run.precision == Precision.bfloat16:
                minibatch_data["image"] = minibatch_data["image"].bfloat16()

            if self.args.run.precision == Precision.mixed:
                minibatch_data["image"] = minibatch_data["image"].half()

            if self.args.run.compute_mode == ComputeMode.XPU:
                if self.args.data.data_format == DataFormatKind.channels_last:
                    minibatch_data["image"] == minibatch_data['image'].to(memory_format=torch.   channels_last)
                    minibatch_data["label"] == minibatch_data['label'].to(memory_format=torch.channels_last)
```
to support different data types. This could be done through a separate data loader in standard ways, 
we are just showing the specific way we did that.

Then we select different batch size and parameters. We present the best result out of 3-5 repeatations 
of a parameter set. 

To perform the single GPU calcualtion, we run 2 MPI ranks on the GPU, as there are 2-tiles. To 
specifically target 2 tiles from the same GPU, we have to specify the `export ZE_AFFINITY_MASK=0.0,0.
1` parameter. This example targets the two tiles of the first GPU -- denoted by 0.0 and 0.1.

Another important environment flag is `export IPEX_FP32_MATH_MODE=TF32` which turns the TF32 mode on 
for Intel GPUs.

As we wanted to focus on the GPU performances, we by hand put `export NUMEXPR_MAX_THREADS=1` to only 
assign 1 CPU thread per GPU process.

Now we look at a typical ouput generated for such a run. We have presented results for 500 iterations, 
for demonstration we do only a 100.

## Results: 
Now that we have showed how we may run a particular application, we can look what we get out of this 
excercise. We have done training with these applications with three precision types FP32 (default on 
Intel, AMD), TF32 (default on Nvidia), and reduced precision (BF16 for Intel and FP16 for Nvidia, AMD).

We wanted to look at the application performance at the level of a single GPU and a Single Node. We 
will discuss the important bits of the results at the Single GPU level, as we have found similar 
trends to follow in the full node runs as well.

This slide shows the compilation of the results, and pretty busy. We include this for reference, as 
the slides will be made available. The legends here is that colors represents devices, and hatches 
represents precision types.

We observe that TF32, in general, yields higher throughput.

Reduced precision can help, but it is highly dependent on the application. Although, in more than 50% 
of the cases we have seen performance improvement (marked with yellow lightnings), there are quite a 
few cases where we have seen similar throughput (marked with black approximate signs). In a couple of 
occasions we have seen performance regression (marked with red snails).

For CT with TF implementation, XLA kernel fusion provides significant performance improvements, 
specially with reduced precision on A100.

Frameworks can have an impact on performance as well. We explored CT with both PT and TF 
implementations. We have found that PT in general performs better than TF, at least at the single GPU 
and single node level. The performance difference are specially prominent on the Intel device. 

## Conclusion slide:
To conclude, I will re-iterate the key take home messages:
- AI applications written with Python are fairly portable and performant with minimal changes to the 
source code.
- For smaller models, using multiple ranks per device is likely to increase throughput.

A couple of technical take aways:
- TF32 in general yields better throughput
- XLA fusion of computational kernels are quite effective
- Reduced precision training can be benificial, but it is highly model dependent. 