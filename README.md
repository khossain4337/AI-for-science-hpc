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
For most of our applications, we followed the strategy of ```pip``` installing the application dependencies on a ```conda``` environment. 

