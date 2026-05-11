# **Spike-Driven-Transformer+Spik4lite Demo**  

## Introduction
This is a demo implementation of "**Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices**".
The demo follows the **train-and-infer** pipeline to show the basic usage of Spike-Driven-Transformer+Spik4lite.

## Directory

The `Spike-Driven-Transformer` is the root and its structure is described as below.

```
│Spike-Driven-Transformer/
├──README.md
│  ├──cifar10
│  │   ├──cifar10.yml
│  │   ├──test.py
│  │   ├──train.log
│  │   ├──train.py
│  ├──cifar10-dvs
│  │   ├──CIFAR10DVS.yml
│  │   ├──test.py
│  │   ├──train.log
│  │   ├──train.py
│  ├──cifar100
│  │   ├──cifar100.yml
│  │   ├──SOPs_consumption_on_cifar100.py
│  │   ├──test.py
│  │   ├──train.py
│  ├──dvs-gesture
│  │   ├──autoaugment.py
│  │   ├──SDT  Power profile.csv
│  │   ├──SDT+Spik4lite Power profile.csv
│  │   ├──SOPs_consumption_on_dvs-gesture.py
│  │   ├──test.py
│  │   ├──train.py
│  │   ├──utils.py
│  ├──dvs_utils
│  ├──model
│  ├──module
│  ├──criterion.py
```
* `README.md` provides the general overview and introduction of the project, as well as instructions on how to run the code.

## Running the Demo
The whole pipeline contains three major steps: (1) activate the Conda Environment, (2) Run the train code and Run the SOPs consumption code.
### (1) activate the Conda Environment
```
cd Spike-Driven-Transformer
```
```
conda activate Spik4lite
```
### (2) Run the code
#### Runing  on CIFAR10
Setting hyper-parameters in cifar10.yml

```
python cifar10/train.py
```
#### Runing  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
python cifar100/train.py
```
#### Runing  on CIFAR10-DVS
Setting hyper-parameters in CIFAR10DVS.yml
```
python cifar10-dvs/train.py
```
#### Runing  on DVS128 Gesture
```
python dvs-gesture/train.py
```
#### Runing  on SOPs consumption

```
python SOPs_consumption_on_dvs-gesture.py
```