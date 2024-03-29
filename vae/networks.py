#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:11:59 2019
networks.py 
a selection of networks to use in the model
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
import tensorflow_hub as hub


def conv_encoder(params, is_training=True):

    activation = params['activation']
    latent_size= params['latent_size']
    n_filt = params['n_filt']
    bias   = params['bias']
    dataset = params['data_set']

    def encoder(x):
        with tf.variable_scope('model/encoder',['x'], reuse=tf.AUTO_REUSE):

            net = tf.layers.conv2d(x,n_filt,5,strides=2,activation=None, padding='SAME', use_bias=bias) #64x64 -> 32x32/ 28x28 -> 14x14
            net = tf.layers.batch_normalization(net, training=is_training)
            net = activation(net)

            net = tf.layers.conv2d(net, n_filt*2, 5, 2, activation=None, padding='SAME', use_bias=bias) #32x32 -> 16x16 / 14x14 -> 7*7
            net = tf.layers.batch_normalization(net, training=is_training)
            net = activation(net)

            if dataset in ['celeba']:
                net = tf.layers.conv2d(net, n_filt*4, 5, 2, activation=None, padding='SAME', use_bias=bias) #16x16 -> 8x8 
                net = tf.layers.batch_normalization(net, training=is_training)
                net = activation(net)

            net = tf.layers.flatten(net) #8x8*n_filt*4
            net = tf.layers.dense(net, latent_size*2, activation=None)
            return net

    return encoder


def conv_decoder(params, is_training=True):

    activation = params['activation']
    latent_size= params['latent_size']
    n_filt = params['n_filt']
    bias   = params['bias']
    dataset = params['data_set']

    def decoder(z):
        with tf.variable_scope('model/decoder',['z'], reuse=tf.AUTO_REUSE):

            if dataset in ['celeba','cifar10']:
                NN = 8
            elif dataset in ['mnist','fmnist']: 
                NN = 7
            net = tf.layers.dense(z,n_filt*4*NN*NN,activation=activation, use_bias=bias)
            net = tf.reshape(net, [-1, NN, NN,n_filt*4])

            if dataset in ['celeba']:
                net = tf.layers.conv2d_transpose(net,n_filt*4, 5, strides=2, padding='SAME', use_bias=bias) # output_size 16x16/14x14
                net = tf.layers.batch_normalization(net, training=is_training)
                net = activation(net)

            net = tf.layers.conv2d_transpose(net,n_filt*2, 5, strides=2, padding='SAME', use_bias=bias) # output_size 16x16/14x14 
            net = tf.layers.batch_normalization(net, training=is_training)
            net = activation(net)

            net = tf.layers.conv2d_transpose(net, n_filt, 5, strides=2, padding='SAME', use_bias=bias) # output_size 32x32/28x28
            net = tf.layers.batch_normalization(net, training=is_training)
            net = activation(net)

            net = tf.layers.conv2d_transpose(net, params['output_size'][-1], kernel_size=4, strides=1, activation=None, padding='same', name='output_layer')# bring to correct number of channels
        return net

    return decoder

def fully_connected_encoder(params,is_training):

    activation = params['activation']
    latent_size = params['latent_size']

    def encoder(x):
        with tf.variable_scope('model/encoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(x, 512, name='dense_1', activation=activation)
            net = tf.layers.dense(net, 256, name='dense_2', activation=activation)
            if params['dropout']:
                if is_training:
                    net = tf.nn.dropout(net,rate=params['rate'])
            net = tf.layers.dense(net, 128, name='dense_3', activation=activation)
            net = tf.layers.dense(net, 2*latent_size, name='dense_4', activation=None)
        return net
    return encoder 

def fully_connected_decoder(params,is_training):
    
    activation = params['activation']
    latent_size= params['latent_size']

    def decoder(z):
        with tf.variable_scope('model/decoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(z, 128, name ='dense_1', activation=activation)
            net = tf.layers.dense(net, 256, name='dense_2', activation=activation)
            if params['dropout']:
                if is_training:
                    net = tf.nn.dropout(net, rate=params['rate'])
            net = tf.layers.dense(net, 512, name='dense_3', activation=activation)
            net = tf.layers.dense(net, params['output_size'] , name='dense_4', activation=None)
        return net
    return decoder


def make_encoder(params, is_training):
    
    network_type = params['network_type']

    if network_type=='fully_connected':
        encoder_ = fully_connected_encoder(params,is_training)
    elif network_type=='conv':
        encoder_ = conv_encoder(params,is_training)
    else:
        raise NotImplementedError("Network type not implemented.")

    def encoder_spec():
        x = tf.placeholder(tf.float32, shape=params['full_size'])
        z = encoder_(x)
        hub.add_signature(inputs={'x':x},outputs={'z':z})

    enc_spec  = hub.create_module_spec(encoder_spec)

    encoder   = hub.Module(enc_spec, name='encoder',trainable=True)

    hub.register_module_for_export(encoder, "encoder")

    return encoder


def make_decoder(params,is_training):

    network_type = params['network_type']

    if network_type=='fully_connected':
        decoder_ = fully_connected_decoder(params,is_training)
    elif network_type=='conv':
        decoder_ = conv_decoder(params, is_training)
    else:
        raise NotImplementedError("Network type not implemented.")

    def decoder_spec():
        z = tf.placeholder(tf.float32, shape=[None,params['latent_size']]) 
        x = decoder_(z)
        hub.add_signature(inputs={'z':z},outputs={'x':x})

    dec_spec  = hub.create_module_spec(decoder_spec)

    decoder   = hub.Module(dec_spec, name='decoder',trainable=True)

    hub.register_module_for_export(decoder, "decoder")

    return decoder

