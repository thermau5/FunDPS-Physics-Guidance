# FunctionSpaceDiffusion

## Setup

To install the required packages, run:

```shell
conda env create -f environment.yml
```

Please follow the instructions in [neuraloperator](https://github.com/neuraloperator/neuraloperator) to install the library v1.0.1.

Please follow the instructions in [DiffusionPDE](https://github.com/jhhuangchloe/DiffusionPDE) to download the data and place it in the `data/DiffPDE` directory.

To generate the data, run:

```shell
python utils/dataset_process.py all
```

To initialize the wandb environment, run:

```shell
wandb init
```

## Usage

```shell
# Train a new diffusion model on the Darcy Flow dataset.
python train.py -c=configs/training/darcy.yml --name=darcy-test

# Recover both spaces with observation on both sides
python generate_pde.py --config configs/generation/darcy.yaml
```

## Contributing

Please apply the `black` code formatter with the `--line-length=200` option. We typically wouldn't opt for such a long line length, but due to numerous existing lengthy lines, this will help reduce discrepancies.
