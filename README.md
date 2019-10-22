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
 
## How to use this repo?
 
To install, download the repo and run 
```
pip install -e .
```
To see all available settings for running the VAE

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
![recons](/plots/vae_recons.png)

![samples](/plots/vae_samples.png)

