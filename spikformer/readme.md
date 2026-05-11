# Spikformer + Spik4lite Demo

This folder provides the **Spikformer + Spik4lite** demo for the paper **"Spik4lite: Refactoring Neuromorphic Sparsity for Efficient Spiking Neural Networks on Commodity Edge Devices"**.

Each dataset folder is self-contained and includes the model definition, training script, testing script, and dataset-specific helper files.

## Directory

```text
spikformer/
|-- README.md
|-- cifar10/
|   |-- aa_snn.py
|   |-- cifar10.yml
|   |-- loader.py
|   |-- model.py
|   |-- train.py
|   |-- test.py
|   `-- transforms_factory.py
|-- cifar100/
|   |-- aa_snn.py
|   |-- cifar100.yml
|   |-- loader.py
|   |-- model.py
|   |-- train.py
|   |-- test.py
|   `-- transforms_factory.py
|-- cifar10dvs/
|   |-- autoaugment.py
|   |-- model.py
|   |-- train.py
|   |-- test.py
|   `-- utils.py
`-- dvs128gesture/
    |-- autoaugment.py
    |-- model.py
    |-- train.py
    |-- test.py
    `-- utils.py
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

Run commands from the `spikformer` directory unless otherwise noted:

```bash
cd spikformer
conda activate Spik4lite
```

### CIFAR-10

Set hyperparameters in `cifar10/cifar10.yml`, then run:

```bash
cd cifar10
python train.py
python test.py
```

### CIFAR-100

Set hyperparameters in `cifar100/cifar100.yml`, then run:

```bash
cd cifar100
python train.py
python test.py
```

### CIFAR10-DVS

```bash
cd cifar10dvs
python train.py
python test.py
```

### DVS128 Gesture

```bash
cd dvs128gesture
python train.py
python test.py
```

## Notes

- Update dataset paths and training hyperparameters before running each experiment.
- This folder does not include separate SOPs scripts; use the training and testing scripts for the Spikformer demo workflow.
- Directory names for event datasets do not contain hyphens in this baseline: `cifar10dvs` and `dvs128gesture`.
