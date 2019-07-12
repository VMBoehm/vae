#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:29:59 2019
model.py 
builds the vae model
@author: nessa
"""

import tensorflow as tf
import networks as nw

def model_fn(features, labels, mode, params, config):
    del labels, config
    
    #putting fully connected stuff for mnist here, but should be generalized
    encoder      = nw.make_encoder(params['activation'], params['latent_size'], params['network_type'])
  
    decoder      = nw.make_decoder(params["activation"], params["output_size"], params['network_type'])
    
    latent_prior = make_prior(params["latent_size"])
    
    approx_posterior = encoder(features)

    approx_posterior_sample = approx_posterior.sample(params["n_samples"])
    decoder_likelihood = decoder(approx_posterior_sample)
  
    neg_log_likeli = - decoder_likelihood.log_prob(features)
    avg_log_likeli = tf.reduce_mean(input_tensor=neg_log_likeli)
    tf.summary.scalar("log_likelihood", avg_log_likeli)
  
    kl = tfd.kl_divergence(approx_posterior, latent_prior)
    avg_kl = tf.reduce_mean(kl)
    tf.summary.scalar("KL_divergence", avg_kl)
  
    elbo_local = -(kl+neg_log_likeli)
    elbo       = tf.reduce_mean(elbo_local)
  
    tf.summary.scalar("elbo", elbo)
  
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
