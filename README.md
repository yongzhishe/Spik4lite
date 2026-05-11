# Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices

This repository provides a demo implementation for the paper **"Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices"**.

Spik4lite is designed as a lightweight plug-and-play module for improving the accuracy-efficiency trade-off of spiking neural networks (SNNs), especially SNN-Transformer models deployed on commodity edge devices such as NVIDIA Jetson platforms.

---

## Abstract

Modern SNNs, especially transformer-based architectures, can benefit from sparse spike-driven computation. In practice, however, commodity edge devices often cannot fully exploit theoretical neuromorphic sparsity, which may lead to unnecessary computation, latency, and energy cost.

Spik4lite addresses this gap by refactoring channel-wise neuromorphic sparsity. It introduces an energy-aware gating mechanism to identify low-efficiency channels during training, then physically removes redundant channels while compensating for the eliminated spikes. This turns an SNN into a more compact model that can reduce real computation and energy consumption on commodity hardware while preserving model accuracy.

The demo integrates Spik4lite into representative SNN-Transformer baselines, including **Spikformer**, **Spike-Driven Transformer**, and **Spikingformer**, and covers both static image datasets and neuromorphic event datasets.

## Framework

The core pipeline of Spik4lite contains three stages:

1. Train an SNN baseline with learnable energy-aware gates.
2. Accumulate channel statistics and determine the retained channels.
3. Prune the network into a physically compact model for efficient inference.

The repository also includes two paper figures for reference:

- `overview.pdf`: overview of the Spik4lite compression pipeline.
- `model.pdf`: detailed design of the Spik4lite module.

## Getting Started

### 1. Installation

Clone this repository and enter the project directory:

```bash
git clone <your-repository-url>
cd Spik4lite
```

Create the Conda environment from the provided environment file:

```bash
conda env create -f environment.yml
conda activate Spik4lite
```

Main tested dependencies:

- Anaconda
- Python 3.12.2 on GPU server / Python 3.10 on Jetson
- PyTorch 2.5.0
- CUDA 12.4
- SpikingJelly 0.0.0.0.14
- JetPack 6.2.0 (L4T 36.4.3) for Jetson experiments

### 2. Dataset Preparation

The demo covers the following datasets:

- CIFAR-10
- CIFAR-100
- CIFAR10-DVS
- DVS128 Gesture

Please download the datasets and update the corresponding dataset paths or configuration files in each model directory before running experiments. Each baseline folder contains its own README and dataset-specific training scripts.

## Running Experiments

Each baseline directory is self-contained. Enter the model directory first, activate the Conda environment, then run the dataset-specific training script.

### Spikingformer + Spik4lite

```bash
cd Spikingformer
conda activate Spik4lite

cd cifar10
python train.py
```

Other available tasks are located in:

- `Spikingformer/cifar100`
- `Spikingformer/cifar10-dvs`
- `Spikingformer/dvs128-gesture`

### Spikformer + Spik4lite

```bash
cd spikformer
conda activate Spik4lite

cd cifar10
python train.py
```

Other available tasks are located in:

- `spikformer/cifar100`
- `spikformer/cifar10dvs`
- `spikformer/dvs128gesture`

### Spike-Driven Transformer + Spik4lite

```bash
cd Spike-Driven-Transformer
conda activate Spik4lite

python cifar10/train.py
```

Other available tasks are located in:

- `Spike-Driven-Transformer/cifar100`
- `Spike-Driven-Transformer/cifar10-dvs`
- `Spike-Driven-Transformer/dvs-gesture`

### Energy and SOPs Evaluation

Some experiment folders include scripts for SOPs or energy-consumption analysis, for example:

```bash
python SOPs_consumption_on_cifar10.py
```

Please run these scripts from the corresponding dataset directory, following the dedicated README inside each baseline folder.

## Project Structure

```text
Spik4lite/
|-- Spik4lite.py                  # Core Spik4lite layers, gating, and pruning utilities.
|-- environment.yml               # Conda environment used by the demo.
|-- overview.pdf                  # Overview figure of the compression pipeline.
|-- model.pdf                     # Detailed figure of the Spik4lite module.
|-- Spikingformer/                # Spikingformer baseline with Spik4lite demos.
|   |-- cifar10/
|   |-- cifar100/
|   |-- cifar10-dvs/
|   `-- dvs128-gesture/
|-- spikformer/                   # Spikformer baseline with Spik4lite demos.
|   |-- cifar10/
|   |-- cifar100/
|   |-- cifar10dvs/
|   `-- dvs128gesture/
`-- Spike-Driven-Transformer/     # Spike-Driven Transformer baseline with Spik4lite demos.
    |-- cifar10/
    |-- cifar100/
    |-- cifar10-dvs/
    `-- dvs-gesture/
```

## Notes

- The root README provides the overall project entry point.
- For exact hyperparameters, dataset paths, and dataset-specific commands, please refer to the README and configuration files inside each baseline directory.
- Power profiles and SOPs scripts are included for selected experiments to support edge-device efficiency analysis.

## Acknowledgement

This demo builds on representative SNN-Transformer baselines, including Spikformer, Spike-Driven Transformer, and Spikingformer. We thank the original authors for releasing their implementations.
