# VAE
### a tensorflow implementation of a vanilla Variational AutoEncoder (VAE) [[1](https://arxiv.org/abs/1312.6114),[2](https://arxiv.org/abs/1401.4082)] with support for several standard datasets

## What is supported?

### Datasets
 - mnist
 - fashion-mnist
 - cifar10 (conv network only)
 - celeb-a (conv network only)
 
 ### Likelihood types
 - Bernoulli
 - Gaussian
 
 ### Network-types
 - fully connected
 - convolutional
 
 ## How to use this repo?
 
Download the repo and run 
```
python pip . 
```
For instructions for running the VAE with different settings run

```
python main.py --helpfull
```

For example
```
python main.py --data_set='mnist' --latent_size=8 --network_type='conv'
```
will run a VAE on the mnist dataset with a latnt space dimensionality of 8 and (de)convolutional networks as generator/encoder. 

