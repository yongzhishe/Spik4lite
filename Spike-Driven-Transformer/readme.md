# Spike-Driven Transformer + Spik4lite Demo

This folder provides the **Spike-Driven Transformer + Spik4lite** demo for the paper **"Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices"**.

The code follows a train-and-evaluate workflow and includes dataset-specific scripts for CIFAR, CIFAR10-DVS, and DVS128 Gesture experiments.

## Directory

```text
Spike-Driven-Transformer/
|-- README.md
|-- criterion.py
|-- module/
|-- model/
|-- dvs_utils/
|-- cifar10/
|   |-- cifar10.yml
|   |-- train.py
|   `-- test.py
|-- cifar100/
|   |-- cifar100.yml
|   |-- train.py
|   |-- test.py
|   `-- SOPs_consumption_on_cifar100.py
|-- cifar10-dvs/
|   |-- CIFAR10DVS.yml
|   |-- train.py
|   `-- test.py
`-- dvs-gesture/
    |-- autoaugment.py
    |-- train.py
    |-- test.py
    |-- utils.py
    |-- SOPs_consumption_on_dvs-gesture.py
    |-- SDT  Power profile.csv
    `-- SDT+Spik4lite Power profile.csv
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

Run commands from the `Spike-Driven-Transformer` directory unless otherwise noted:

```bash
cd Spike-Driven-Transformer
conda activate Spik4lite
```

### CIFAR-10

Set hyperparameters in `cifar10/cifar10.yml`, then run:

```bash
python cifar10/train.py
python cifar10/test.py
```

### CIFAR-100

Set hyperparameters in `cifar100/cifar100.yml`, then run:

```bash
python cifar100/train.py
python cifar100/test.py
```

To evaluate SOPs for CIFAR-100:

```bash
cd cifar100
python SOPs_consumption_on_cifar100.py
```

### CIFAR10-DVS

Set hyperparameters in `cifar10-dvs/CIFAR10DVS.yml`, then run:

```bash
python cifar10-dvs/train.py
python cifar10-dvs/test.py
```

### DVS128 Gesture

```bash
python dvs-gesture/train.py
python dvs-gesture/test.py
```

To evaluate SOPs for DVS128 Gesture:

```bash
cd dvs-gesture
python SOPs_consumption_on_dvs-gesture.py
```

## Notes

- Update dataset paths and training hyperparameters before running each experiment.
- `model/`, `module/`, `criterion.py`, and `dvs_utils/` provide shared components used by the dataset-specific scripts.
- The CSV power profiles are provided for selected edge-device energy measurements.
