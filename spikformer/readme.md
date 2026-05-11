# **spikformer+Spik4lite Demo**  

## Introduction
This is a demo implementation of "**Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices**".
The demo show the basic usage of spikformer+Spik4lite.

## Directory

The `spikformer` is the root and its structure is described as below.

```
│spikformer/
├──README.md
│  ├──cifar10
│  │   ├──aa_snn.py
│  │   ├──cifar10.yml
│  │   ├──loader.py
│  │   ├──model.py
│  │   ├──test.py
│  │   ├──train.py
│  │   ├──transforms_factory.py
│  ├──cifar10dvs
│  │   ├──autoaugment.py
│  │   ├──model.py
│  │   ├──test.py
│  │   ├──train.log
│  │   ├──train.py
│  │   ├──utils.py
│  ├──cifar100
│  │   ├──aa_snn.py
│  │   ├──cifar100.yml
│  │   ├──loader.py
│  │   ├──model.py
│  │   ├──test.py
│  │   ├──train.log
│  │   ├──train.py
│  │   ├──transforms_factory.py
│  ├──dvs128gesture
│  │   ├──autoaugment.py
│  │   ├──model.py
│  │   ├──test.py
│  │   ├──train.py
│  │   ├──utils.py
```
* `README.md` provides the general overview and introduction of the project, as well as instructions on how to run the code.

## Running the Demo
The whole pipeline contains three major steps: (1) activate the Conda Environment, (2) Run the train code.
### (1) activate the Conda Environment
```
cd spikformer
```
```
conda activate Spik4lite
```
### (2) Run the code
#### Runing  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

#### Runing  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar100
python train.py
```

#### Runing  on DVS128 Gesture
```
cd dvs128gesture
python train.py
```
#### Runing  on CIFAR10-DVS
```
cd cifar10dvs
python train.py
```