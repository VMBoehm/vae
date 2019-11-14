# VAE
### a tensorflow implementation of a vanilla Variational AutoEncoder (VAE) [[1](https://arxiv.org/abs/1312.6114),[2](https://arxiv.org/abs/1401.4082)] with support for several standard datasets

## What is supported?

### Datasets
 - [mnist](http://yann.lecun.com/exdb/mnist/) 
 - [fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)
 - [cifar10](https://www.cs.toronto.edu/~kriz/cifar.html) (convolutional network only)
 - [celeb-a](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (convolutional network only)
 
### Likelihood types
- Bernoulli
- Gaussian
 
### Network types
- fully connected
- convolutional

### Misc
 - The VAE can be trained on subsets with a specific label by specifying the 'class_label' flag.
 - The VAE can be trained without the 'V' that is as an AutoEncoder by setting the 'AE' flag.
 - there's a number of regularization options available for the fully connected network (dropout, L2 regularization on network weights)
 - data augmentation is posisble by adding noise or image rotations

## Installation
 
To install, either download the repo and run 
```
pip install -e .
```
or run 
```
pip install git+https://github.com/VMBoehm/vae
```

## Usage

To see all available settings, simply run

```
python main.py --helpfull
```

### Example
```
python main.py --data_set='mnist' --latent_size=8 --network_type='conv'
```
will run a VAE on the mnist dataset with a latent space dimensionality of 8 and (de)convolutional networks as generator/encoder.

### Outputs

The code automatically saves checkpoints and exports the trained model. During the training, several summaries can be visualized with tensorboard:

```
tensorboard --logdir='./model/'
```

e.g. mnist reconstructions with a Gaussian likelihood, a latent space dimensionality of 8 and a fully connected network

![recons](/plots/vae_recons.png)  

and corresponding samples  

![samples](/plots/vae_samples.png)  

## TF2 compatibility

A tensorflow 2.0 compatible version is available in the tf2 branch

