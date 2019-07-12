#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:11:59 2019
networks.py 
a selection of networks to use in the model
@author: nessa
"""

import tensorflow as tf

def fully_connected_encoder(activation,latent_size):

    def encoder(x):
        with tf.variable_scope('model/encoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(x, 512, name='dense_1', activation=activation)
            net = tf.layers.dense(net, 384, name='dense_2', activation=activation)
            net = tf.layers.dense(net, 256, name='dense_3', activation=activation)
            net = tf.layers.dense(net, 2*latent_size, name='dense_4', activation=None)
        return net

    return encoder 

def fully_connected_decoder(activation, output_size):
    
    def decoder(z)
        with tf.variable_scope('model/decoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(z, 256, name='dense_1', activation=activation)
            net = tf.layers.dense(net, 384, name='dense_2', activation=activation)
            net = tf.layers.dense(net, 512, name='dense_3', activation=activation)
            net = tf.layers.dense(net, output_size , name='dense_4', activation=None)
        return net
    return decoder

def make_encoder(activation, latent_size, network_type):
    
    if network_type=='fully_connected':
        encoder = fully_connected_encoder(activation, latent_size)
    else:
        raise NotImplementedError("Network type not implemented.")
    
    return encoder


def make_decoder(activation,latent_size,network_type):

    if network_type=='fully_connected':
        decoder = fully_connected(activation, output_size)
    else:
        raise NotImplementedError("Network type not implemented.")

    return decoder

