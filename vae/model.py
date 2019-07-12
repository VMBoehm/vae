#!/usr/bin/env python3
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

def get_prior(latent_size):
    return tfd.MultivariateNormalDiag(tf.zeros(latent_size), scale_identity_multiplier=1.0)

def get_posterior(encoder):

    def posterior(x):

        mu, sigma        = tf.split(encoder(x), 2, axis=-1)
        sigma            = tf.nn.softplus(sigma) + 0.0001
        approx_posterior = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sigma)
        
        return approx_posterior

    return posterior

def get_likelihood(decoder, likelihood_type):

    if likelihood_type=='Bernoulli':
        def likelihood(z):
            return tfd.Independent(tfd.Bernoulli(logits=decoder(z)))
    return likelihood

def model_fn(features, labels, mode, params, config):
    del labels, config
    
    #putting fully connected stuff for mnist here, but should be generalized
    encoder      = nw.make_encoder(params['activation'], params['latent_size'], params['network_type'])
    decoder      = nw.make_decoder(params['activation'], params['output_size'], params['network_type'])
    
    posterior    = get_posterior(encoder)
    prior        = get_prior(params['latent_size'])
    likelihood   = get_likelihood(decoder, params['likelihood'])

    approx_posterior        = posterior(features)
    approx_posterior_sample = approx_posterior.sample(params['n_samples'])
    decoder_likelihood      = likelihood(approx_posterior_sample)
    neg_log_likeli = - decoder_likelihood.log_prob(features)
    avg_log_likeli = tf.reduce_mean(input_tensor=neg_log_likeli)

  
    kl             = tfd.kl_divergence(approx_posterior, prior)
    avg_kl         = tf.reduce_mean(kl)
  
    elbo           = -(avg_kl+avg_log_likeli)
  
    tf.summary.scalar("elbo", elbo)
    tf.summary.scalar("log_likelihood", avg_log_likeli)
    tf.summary.scalar("kl_divergence", avg_kl)

    loss = -elbo
  
    global_step   = tf.train.get_or_create_global_step()
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step, params["max_steps"])
    optimizer     = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=global_step)
  
    eval_metric_ops={
        'elbo': tf.metrics.mean(elbo),
        'negative_log_likelihood': tf.metrics.mean(avg_log_likeli),
        'kl_divergence': tf.metrics.mean(avg_kl)
    }
  
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops = eval_metric_ops)
