# /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:29:59 2019
model.py 
builds the vae model
@author: nessa
"""

"""
Copyright 2019 Vanessa Martina Boehm

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import vae.networks as nw


### these two functions are inspired by https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py and modified to work with flattened data by adding a shape keyword

def pack_images(images, rows, cols,shape):
    """Helper utility to make a field of images.
    Borrowed from Tensorflow Probability
    """
def make_images(images, nrows, ncols,shape):
    width  = shape[-3]
    height = shape[-2]
    depth  = shape[-1]
    bsize  = tf.shape(input=images)[0]
    images = tf.reshape(images, (-1, width, height, depth))
    nrows  = tf.minimum(nrows, bsize)
    ncols  = tf.minimum(bsize//nrows, ncols)
    images = images[:nrows * ncols]
    images = tf.reshape(images, (nrows, ncols, width, height, depth))
    images = tf.transpose(a=images, perm=[0, 2, 1, 3, 4])
    images = tf.clip_by_value(tf.reshape(images, [1, nrows * width, ncols * height, depth]), 0, 1)
    return images


def image_tile_summary(name, tensor, rows, cols, shape):
    tf.compat.v1.summary.image(name, make_images(tensor, rows, cols, shape), max_outputs=1)
#######

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

        with tf.compat.v1.variable_scope("likelihood", reuse=tf.compat.v1.AUTO_REUSE):
            sigma = tf.compat.v1.get_variable(name='sigma', initializer=sig)
        tf.compat.v1.summary.scalar('sigma', sigma)

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

    encoder      = nw.make_encoder(params, is_training)
    decoder      = nw.make_decoder(params, is_training)

    posterior               = get_posterior(encoder)
    approx_posterior        = posterior(features)

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        prior        = get_prior(params['latent_size'])
        likelihood   = get_likelihood(decoder, params['likelihood'], params['sigma'])

        approx_posterior_sample = approx_posterior.sample()
        decoder_likelihood      = likelihood(approx_posterior_sample)
        prior_sample            = prior.sample(params['batch_size'])
        decoded_samples         = likelihood(prior_sample).mean()

        if mode == tf.estimator.ModeKeys.TRAIN:
            image_tile_summary('training/inputs',features, rows=4, cols=4, shape=params['data_shape'])
            image_tile_summary('training/reconstructions',decoder_likelihood.mean(), rows=4, cols=4, shape=params['data_shape'])
        else:
            image_tile_summary('test/inputs',features, rows=4, cols=4, shape=params['data_shape'])
            image_tile_summary('test/reconstructions',decoder_likelihood.mean(), rows=4, cols=4, shape=params['data_shape'])
 
        image_tile_summary('samples',decoded_samples, rows=4, cols=4, shape=params['data_shape'])  
       
        neg_log_likeli  = - decoder_likelihood.log_prob(features)
        avg_log_likeli  = tf.reduce_mean(input_tensor=neg_log_likeli)

        global_step   = tf.compat.v1.train.get_or_create_global_step()
        if params['annealing']:
            beta = 1.-tf.compat.v1.train.cosine_decay(1., global_step, params["max_steps"]//2)
        elif params['AE']:
            beta = 0.
        else:
            beta = 1.

        kl             = tfd.kl_divergence(approx_posterior, prior)
        avg_kl         = tf.reduce_mean(input_tensor=kl)
  
        elbo           = -(beta*avg_kl+avg_log_likeli)
  
        tf.compat.v1.summary.scalar("negative_log_likelihood", avg_log_likeli)
 
        tf.compat.v1.summary.scalar("kl_divergence", avg_kl)
        tf.compat.v1.summary.scalar("elbo", elbo)

        if params['L2-reg']:
            tvars  = tf.compat.v1.trainable_variables()
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tvars if ('bias' not in v.name and 'sigma' not in v.name)]) * 0.01
            tf.compat.v1.summary.scalar("L2_loss",lossL2)
        else:
            lossL2 = 0.

        loss   = -elbo+lossL2

        if params['schedule']==True:
            learning_rate = tf.compat.v1.train.cosine_decay(params["learning_rate"], global_step, params["max_steps"])
        else:
            learning_rate = params['learning_rate']

        optimizer     = tf.compat.v1.train.AdamOptimizer(learning_rate)

        tf.compat.v1.summary.scalar('learning_rate',learning_rate)
        tf.compat.v1.summary.scalar('beta',beta)

        train_op = optimizer.minimize(loss, global_step=global_step)
    
        eval_metric_ops={
                'elbo': tf.compat.v1.metrics.mean(elbo),
                'negative_log_likelihood': tf.compat.v1.metrics.mean(avg_log_likeli),
                'kl_divergence': tf.compat.v1.metrics.mean(avg_kl),
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops = eval_metric_ops)
    else:
        predictions = {'code': approx_posterior.mean()}
    
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
