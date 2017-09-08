# rl-nn-benchmarks
Common neural networks for reinforcement learning timed under various python frameworks.

### Why?

Reinforcement learning (RL) typically uses different neural networks than supervised learning, in particular much smaller than for image classification.  Although the training regime is similar (minibatches or batches), the networks are often run forward in very small batches or even over a single input.  It is not guaranteed that speed comparisons from supervised models will carry over to RL.

### Contents

Basic results are recorded here.  See the bottom for installation procedures and version information.

Contributions to improve the implementations or add new ones are welcome!

# Results

So far, the basic result is that Theano outperforms all other frameworks by a wide margin.  (Perhaps I have not properly implemented with the others?)


## Atari CNN

Ran on a GTX-1080 (PCIe x16), with Intel i7-6900K.  GPU utilzation observed by eye using `watch -n 0.1 nvidia-smi`.  All values roughly averaged over multiple runs.

Item | Setting
--- | ---
Input Element| 4x84x84 float32 tensor
Inference batch size | 16
Training batch size | 128
Dataset size | 50 training batches
No. of repeats (epochs in one timing) | 10

### Small Network

Framework | Inference (s) | Training (s) | Inference GPU Util. (%) | Training GPU Util. (%)
--- | ---:| ---:| ---:| ---:|
Theano | **1.8** | **1.5** | 64 | 91
Tensorflow | 6.0 | 2.9 | 22 | 56
Chainer | 5.2 | 2.8 | 25 | 64
PyTorch | 3.5 | 3.2* | 35 | 47*

(*) wide variance between runs

### Large Network

Framework | Inference (s) | Training (s) | Inference GPU Util. (%) | Training GPU Util. (%)
--- | ---:| ---:| ---:| ---:|
Theano | **2.3** | **2.3** | 71 | 93
Tensorflow | 6.5 | 3.7 | 26 | 62
Chainer | 6.4 | 3.4 | 24 | 71
PyTorch | 4.0 | 3.6* | 39 | 60*

(*) wide variance between runs


# Installation & Versions

All tests ran with the following versions:

Item | Version
--- | ---
Ubuntu | 16.04
CUDA | 8.0
cuDNN | 6
Python | 3.5
numpy | 1.13
Theano | 0.9.0
libgpuarray / pygpu | 0.6.9
Tensorflow | 1.3.0
Chainer | 2.0.2
cupy | 1.0.2
PyTorch | 0.2.0_4


Python installations managed in conda (as of 08-Septemeber-2017):

```
$ conda update conda
# Repeat these or clone for each framework:
$ conda create -n <framework_name> python=3.5 anaconda
$ source activate <framework_name>
(<framework_name>)$ conda install numpy  # (bumps to 1.13, had problems with anaconda install)
```

## Theano Install

```
$ source activate theano
(theano)$ conda install theano pygpu
```

And run with the following Theano flags: `THEANO_FLAGS=device=cuda,gpuarray.preallocate=1,floatX=float32`

## Tensorflow Install

```
$ source activate tensorflow
(tensorflow)$ pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp35-cp35m-linux_x86_64.whl
# Or use the link for your desired installation
```

## Chainer Install

```
$ source activate chainer
(chainer)$ pip install cupy
(chainer)$ pip install chainer
```

## PyTorch Install

```
$ source activate pytorch
(pytorch)$ conda install pytorch torchvision cuda80 -c soumith
```
