# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:29:59 2019
model.py 
builds the vae model
@author: nessa
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import vae.networks as nw


### these two function are taken from https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py and modified to work with flattened data ny adding a shape keyword

def pack_images(images, rows, cols,shape):
    """Helper utility to make a field of images.
    Borrowed from Tensorflow Probability
    """
    width  = shape[-3]
    height = shape[-2]
    depth  = shape[-1]
    images = tf.reshape(images, (-1, width, height, depth))
    batch = tf.shape(images)[0]
    rows = tf.minimum(rows, batch)
    cols = tf.minimum(batch // rows, cols)
    images = images[:rows * cols]
    images = tf.reshape(images, (rows, cols, width, height, depth))
    images = tf.transpose(images, [0, 2, 1, 3, 4])
    images = tf.clip_by_value(tf.reshape(images, [1, rows * width, cols * height, depth]), 0, 1)
    return images

def image_tile_summary(name, tensor, rows, cols, shape):
    tf.summary.image(name, pack_images(tensor, rows, cols, shape), max_outputs=1)

#############

def get_prior(latent_size):
    return tfd.MultivariateNormalDiag(tf.zeros(latent_size), scale_identity_multiplier=1.0)

def get_posterior(encoder):

    def posterior(x):
        mu, sigma        = tf.split(encoder({'x':x},as_dict=True)['z'], 2, axis=-1)
        sigma            = tf.nn.softplus(sigma) + 0.0001
        approx_posterior = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        return approx_posterior

    return posterior

def get_likelihood(decoder, likelihood_type, sig):

    if likelihood_type=='Bernoulli':
        def likelihood(z):
            return tfd.Independent(tfd.Bernoulli(logits=decoder({'z':z},as_dict=True)['x']))
 
    if likelihood_type=='Gauss':
        sigma = tf.get_variable(name='sigma', initializer=sig)
        tf.summary.scalar('sigma', sigma)

        def likelihood(z):
            mean = decoder({'z':z},as_dict=True)['x']
            return tfd.Independent(tfd.MultivariateNormalDiag(loc=mean,scale_identity_multiplier=sigma))

    return likelihood

def model_fn(features, labels, mode, params, config):
    del labels, config

    try:
        features = features['x']
    except:
        pass

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #putting fully connected stuff for mnist here, but should be generalized
    encoder      = nw.make_encoder(params, is_training)
    decoder      = nw.make_decoder(params, is_training)

    posterior               = get_posterior(encoder)
    approx_posterior        = posterior(features)

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        prior        = get_prior(params['latent_size'])
        likelihood   = get_likelihood(decoder, params['likelihood'], params['sigma'])

        image_tile_summary('inputs',features, rows=4, cols=4, shape=params['image_shape'])

        approx_posterior_sample = approx_posterior.sample()
        decoder_likelihood      = likelihood(approx_posterior_sample)

        prior_sample    = prior.sample(params['batch_size'])
        decoded_samples = likelihood(prior_sample).mean()

        image_tile_summary('recons',decoder_likelihood.mean(), rows=4, cols=4, shape=params['image_shape'])
        image_tile_summary('samples',decoded_samples, rows=4, cols=4, shape=params['image_shape'])  
       
        neg_log_likeli  = - decoder_likelihood.log_prob(features)
        avg_log_likeli  = tf.reduce_mean(input_tensor=neg_log_likeli)

        kl             = tfd.kl_divergence(approx_posterior, prior)
        avg_kl         = tf.reduce_mean(kl)
  
        elbo           = -(avg_kl+avg_log_likeli)
  
        tf.summary.scalar("elbo", elbo)
        tf.summary.scalar("log_likelihood", avg_log_likeli)
        tf.summary.scalar("kl_divergence", avg_kl)

        loss          = -elbo
  
        global_step   = tf.train.get_or_create_global_step()
        learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step, params["max_steps"])
        optimizer     = tf.train.AdamOptimizer(learning_rate)

        tf.summary.scalar('learning_rate',learning_rate)

        train_op = optimizer.minimize(loss, global_step=global_step)
  
        eval_metric_ops={
            'elbo': tf.metrics.mean(elbo),
            'negative_log_likelihood': tf.metrics.mean(avg_log_likeli),
            'kl_divergence': tf.metrics.mean(avg_kl)
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops = eval_metric_ops)
    else:
        predictions = {'code': approx_posterior.mean()}
    
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
