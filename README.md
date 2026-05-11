## **Spik4lite**  
### Introduction
This is a demo implementation of "**Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices**".You can find a specific README in each model directory for running instructions.
### Requirements
#### GPU Server:  
* Anaconda
* Python 3.12.2
* PyTorch 2.5.0
* Cuda 12.4
* SpikingJelly 0.0.0.0.14
#### Jetson: 
* Anaconda
* Python 3.10
* PyTorch 2.5.0
* Jetpack 6.2.0 (L4T 36.4.3)
* Spikingjelly 0.0.0.0.14
### Directory
The `demo-code` is the root and its structure is described as below.
```
│SUPPLEMENTARY MATERIAL/
├──Spikingformer/
├──spikformer/
├──Spike-Driven-Transformer/
├──README.md
├──environment.yml
├──Spik4lite.py
```
* `environment.yml`  contains the dependencies to create a Conda environment.
* `Spikingformer`, `spikformer`, and `Spike-Driven-Transformer` correspond to different methods. Each folder is self-contained, including training scripts, energy consumption test files, model definitions, and a dedicated README with detailed instructions.
* `README.md` provides the general overview and introduction of the project.

### Create the Conda Environment
```
cd Supplementary Material
```
```
conda env create -f ./environment.yml
```
```
conda activate Spik4lite
```

