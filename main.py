#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Jul 11 09:32:01 2019
main.py 
run this script to  train your model
@author: nessa
'''

# standard packages
from absl import flags
import numpy as np
import functools
import os

# tensorflow packages
import tensorflow as tf
import tensorflow_hub as hub

import vae.create_datasets as crd
from  vae.model import model_fn


flags.DEFINE_string('model_dir', default='./model', help='directory for storing the model')
flags.DEFINE_string('data_type', default='mnist', help='the tensorflow-dataset to load')
	
flags.DEFINE_float('learning_rate', default=1e-3, help='learning rate')    
flags.DEFINE_integer('batch_size',default=32, help='batch size')
flags.DEFINE_integer('max_steps', default=1000, help='training steps')    
flags.DEFINE_integer('n_steps', default=500, help='number of training steps after which to perfrom the evaluation')

flags.DEFINE_integer('latent_size',default=8, help='dimensionality of latent space')
flags.DEFINE_string('activation', default='leaky_relu', help='activation function')
flags.DEFINE_integer('n_samples', default=16, help='number of samples for encoding')
flags.DEFINE_string('network_type', default='fully_connected', help='whichy type of network to use')

flags.DEFINE_string('likelihood', default='Bernoulli', help='form of likelihood')

flags.DEFINE_integer('class_label', default=-1, help='number of specific class to train on. -1 for all classes')

FLAGS = flags.FLAGS

IMAGE_SHAPES = dict(mnist=[28,28,1],fmnist=[28,28,1],cifar10=[32,32,3])

def main(argv):
    del argv

    params = FLAGS.flag_values_dict()
    IMAGE_SHAPE = IMAGE_SHAPES[FLAGS.data_type]

    params['output_size'] = np.prod(IMAGE_SHAPE)
    params['activation']  = getattr(tf.nn, params['activation'])
    params['width']       = IMAGE_SHAPE[0]
    params['height']      = IMAGE_SHAPE[1]
    params['n_channels']  = IMAGE_SHAPE[2]
    params['image_shape'] = IMAGE_SHAPE

    for dd in ['model_dir']:
        if not os.path.isdir(params[dd]):
            os.mkdir(params[dd])
			
	
    train_input_fn, eval_input_fn = crd.build_input_fns(params['data_type'], params['batch_size'],label=FLAGS.class_label)
    estimator = tf.estimator.Estimator(model_fn, params=params, config=tf.estimator.RunConfig(model_dir=FLAGS.model_dir),)
    
    n_steps = FLAGS.n_steps
    for ii in range(FLAGS.max_steps//n_steps):
        estimator.train(train_input_fn, steps=n_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        print('model evaluation:', eval_results)

    return True

if __name__ == "__main__":
  tf.compat.v1.app.run()
