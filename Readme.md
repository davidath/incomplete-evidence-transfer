### Learning Improved Representations by Transferring Incomplete Evidence Across Heterogeneous Tasks 
Official code repository of ["Learning Improved Representations by Transferring Incomplete Evidence Across Heterogeneous Tasks"](https://arxiv.org/abs/1912.10490). It includes the
implementation of all experiments (using TensorFlow), as well as the scripts used to produce the
figures displayed in the paper. It also includes all neural network
configurations described using [ANNETT-O](https://arxiv.org/abs/1804.02528).
The ANNETT-O description can be found
[here]().

### Setup
```
evitrac/ $ pip install -r requirements.txt
```
### Running experiments

#### Create necessary datasets:
```
evitrac/mkdata/ $ ./mk.py (mnist|cifar|ng20|reuters)
```
#### Create initial latent space:
```
evitrac/ $ ./train.py ini/(mnist|cifar|20ng|reu100k)/px.ini 0
```
#### Create latent categorical evidence variables:
```
evitrac/ $ run/(mnist|cifar|20ng|reu100k)/init_eviae.sh
```
#### mrun scripts were created in order to run in parallel, but can be run sequentially:
```
evitrac/ $ run/(mnist|cifar|20ng|reu100k)/mrun(0|1|2|3).sh
```

#### For triple evidence used in CIFAR-10:
```
evitrac/ $ run/cifar/triple(0|1|2|3).sh
```
