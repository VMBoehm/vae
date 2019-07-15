#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:11:59 2019
networks.py 
a selection of networks to use in the model
@author: nessa
"""

import tensorflow as tf
import tensorflow_hub as hub


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
    
    def decoder(z):
        with tf.variable_scope('model/decoder', reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(z, 256, name='dense_1', activation=activation)
            net = tf.layers.dense(net, 384, name='dense_2', activation=activation)
            net = tf.layers.dense(net, 512, name='dense_3', activation=activation)
            net = tf.layers.dense(net, output_size , name='dense_4', activation=None)
        return net
    return decoder

def make_encoder(activation, output_size, latent_size, network_type):
    
    if network_type=='fully_connected':
        encoder_ = fully_connected_encoder(activation, latent_size)
    else:
        raise NotImplementedError("Network type not implemented.")

    def encoder_spec():
        x = tf.placeholder(tf.float32, shape=[None,output_size])
        z = encoder_(x)
        hub.add_signature(inputs={'x':x},outputs={'z':z})

    enc_spec  = hub.create_module_spec(encoder_spec)

    encoder   = hub.Module(enc_spec, name='encoder',trainable=True)

    hub.register_module_for_export(encoder, "encoder")

    return encoder


def make_decoder(activation,output_size,latent_size,network_type):

    if network_type=='fully_connected':
        decoder_ = fully_connected_decoder(activation, output_size)
    else:
        raise NotImplementedError("Network type not implemented.")

    def decoder_spec():
        z = tf.placeholder(tf.float32, shape=[None,latent_size]) 
        x = decoder_(z)
        hub.add_signature(inputs={'z':z},outputs={'x':x})

    dec_spec  = hub.create_module_spec(decoder_spec)

    decoder   = hub.Module(dec_spec, name='decoder',trainable=True)

    hub.register_module_for_export(decoder, "decoder")

    return decoder

