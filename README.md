# Implicit-Modeling
Unofficial Implement of paper [State-driven Implicit Modeling for Sparsity and Robustness in Neural Networks](https://arxiv.org/pdf/2209.09389.pdf) by Phan Nhat Huy

## To do:
- [x] Complete the overview flow
- [x] Implement parallel SIM training with relax states matching + Cvxpy 
- [x] Sparsity + Robustness optimization
- [ ] Validation with baselines
- [ ] Parallel SIM training with shared memory for X,U
- [ ] ResNet state implementation
- [ ] Quicker solver for implicit model

## Possible featuress:
- [ ] Quick evaluation with other baselines just by configs like SSS, SPR
- [ ] Automatically evaluation and export performance plot
- [ ] A Wrapper for transforming general Pytorch model

## Installation
You should have Mosek license in required path https://docs.mosek.com/10.0/install/license-agreement-info.html
```
python3 -r requirements.txt
```

## How to run
This implementation is fairly simple to use. With a user specificed configuration file, `main.py` will do anything else. The configuration file consists of 3 main parts
: explicit, implicit, and evaluation. Explicit config will declare hyperparameters to generate your "want-to-implicitize" regular deep learning model. 
```yaml
explicit:
  type: fnn
  architecture: [784, 64, 32, 16, 10]
```
It must have its architecture type and specific architecture parameters. And you can Implement your model by yourself in `SIM/explict.py`. It should have 4 methods
`implicit_size(self)` `input_size(self)` and `output_size(self)`. 

Implicit config contain hyperparameters for implicitization like `objective`: `sparsity` or `robustness` and `training_size`.
```yaml
implicit:
  workers: 32
  kappa: 0.99
  objective: sparsity
  lambda1: 0.1
  alpha: 1e-3
  training_size: 0.05 # 0.05% of the training set (can be from 0.0-0.3)
```
Combining all parts we have following general configuration file.
```yaml
explicit:
  type: fnn
  architecture: [784, 64, 32, 16, 10]
  optimizer: Adam
  epoch: 1
  lr: 0.001
  log_interval: 20000
implicit:
  workers: 32
  kappa: 0.99
  objective: sparsity
  lambda1: 0.1
  alpha: 1e-3
  training_size: 0.05 # 0.05% of the training set (can be from 0.0-0.3)
evaluation:
  baselines: [explicit, implicit]
  metric: sparsity
  dataset: FashionMNIST
  bs: 64
```
To run, just input the config path as argument
```
python3 main.py --cfg configs/sparsity.yaml
```