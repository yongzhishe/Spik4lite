# Spikingformer + Spik4lite Demo

This folder provides the **Spikingformer + Spik4lite** demo for the paper **"Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices"**.

The code follows a train-and-evaluate workflow. Each dataset folder contains its own model definition, training script, testing script, and optional SOPs or energy-consumption analysis scripts.

## Directory

```text
Spikingformer/
|-- README.md
|-- energy_consumption_calculation/
|-- cifar10/
|   |-- cifar10.yml
|   |-- model.py
|   |-- train.py
|   |-- test.py
|   |-- SOPs_consumption_on_cifar10.py
|   |-- Spikingformer Power profile.csv
|   `-- Spikingformer+Spik4lite  Power profile.csv
|-- cifar100/
|   |-- cifar100.yml
|   |-- model.py
|   |-- train.py
|   |-- test.py
|   `-- SOPs_consumption_on_cifar100.py
|-- cifar10-dvs/
|   |-- autoaugment.py
|   |-- model.py
|   |-- train.py
|   |-- test.py
|   |-- utils.py
|   `-- SOPs_consumption_on_CIFAR10DVS.py
`-- dvs128-gesture/
    |-- autoaugment.py
    |-- model.py
    |-- train.py
    |-- test.py
    |-- utils.py
    `-- SOPs_consumption_on_DVS128Gesture.py
```

## Environment

Create and activate the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate Spik4lite
```

If the environment already exists, only activate it:

```bash
conda activate Spik4lite
```

## Running Experiments

Run commands from the `Spikingformer` directory unless otherwise noted:

```bash
cd Spikingformer
conda activate Spik4lite
```

### CIFAR-10

Set hyperparameters in `cifar10/cifar10.yml`, then run:

```bash
cd cifar10
python train.py
python test.py
```

To evaluate SOPs for CIFAR-10:

```bash
python SOPs_consumption_on_cifar10.py
```

### CIFAR-100

Set hyperparameters in `cifar100/cifar100.yml`, then run:

```bash
cd cifar100
python train.py
python test.py
```

To evaluate SOPs for CIFAR-100:

```bash
python SOPs_consumption_on_cifar100.py
```

### CIFAR10-DVS

```bash
cd cifar10-dvs
python train.py
python test.py
```

To evaluate SOPs for CIFAR10-DVS:

```bash
python SOPs_consumption_on_CIFAR10DVS.py
```

### DVS128 Gesture

```bash
cd dvs128-gesture
python train.py
python test.py
```

To evaluate SOPs for DVS128 Gesture:

```bash
python SOPs_consumption_on_DVS128Gesture.py
```

## Notes

- Update dataset paths and training hyperparameters before running each experiment.
- The CSV power profiles are provided for selected edge-device energy measurements.
- The `energy_consumption_calculation` folder contains shared utilities for energy and operation analysis.
