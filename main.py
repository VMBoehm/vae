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
flags.DEFINE_string('data_set', default='mnist', help='the tensorflow-dataset to load')
flags.DEFINE_string('module_dir', default='./modules', help='directory to which to export the modules')

flags.DEFINE_float('learning_rate', default=1e-3, help='learning rate')    
flags.DEFINE_integer('batch_size',default=32, help='batch size')
flags.DEFINE_integer('max_steps', default=10000, help='training steps')    
flags.DEFINE_integer('n_steps', default=500, help='number of training steps after which to perfrom the evaluation')

flags.DEFINE_integer('latent_size',default=8, help='dimensionality of latent space')
flags.DEFINE_string('activation', default='leaky_relu', help='activation function')
flags.DEFINE_integer('n_samples', default=16, help='number of samples for encoding')
flags.DEFINE_string('network_type', default='fully_connected', help='whichy type of network to use')

flags.DEFINE_string('likelihood', default='Bernoulli', help='form of likelihood')
flags.DEFINE_float('sigma', default=0.1, help='noise scale used in the Gaussian likelihood')
flags.DEFINE_integer('class_label', default=-1, help='number of specific class to train on. -1 for all classes')

FLAGS = flags.FLAGS

IMAGE_SHAPES = dict(mnist=[28,28,1],fmnist=[28,28,1],cifar10=[32,32,3])

def main(argv):
    del argv

    params = FLAGS.flag_values_dict()
    IMAGE_SHAPE = IMAGE_SHAPES[FLAGS.data_set]

    params['output_size'] = np.prod(IMAGE_SHAPE)
    params['activation']  = getattr(tf.nn, params['activation'])
    params['width']       = IMAGE_SHAPE[0]
    params['height']      = IMAGE_SHAPE[1]
    params['n_channels']  = IMAGE_SHAPE[2]
    params['image_shape'] = IMAGE_SHAPE


    params['model_dir']   = os.path.join(params['model_dir'], '%s'%params['data_set'], '%s'%params['likelihood'], 'class%d'%params['class_label'])
    params['module_dir']   = os.path.join(params['module_dir'], '%s'%params['data_set'], '%s'%params['likelihood'], 'class%d'%params['class_label'])
    
    for dd in ['model_dir', 'module_dir']:
        if not os.path.isdir(params[dd]):
            os.makedirs(params[dd])
			
	
    train_input_fn, eval_input_fn = crd.build_input_fns(params['data_set'], params['batch_size'],label=FLAGS.class_label)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=tf.estimator.RunConfig(model_dir=params['model_dir']),)

    c = tf.placeholder(tf.float32,[params['batch_size'],params['output_size']])
    serving_input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features=dict(x=c))

    #train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=params['max_steps'])
    #eval_spec  = tf.estimator.EvalSpec(input_fn=eval_input_fn)
    exporter   = hub.LatestModuleExporter("tf_hub", serving_input_fn)
    #tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    #exporter.export(estimator, params['module_dir'], estimator.latest_checkpoint())


    n_steps = FLAGS.n_steps
    for ii in range(FLAGS.max_steps//n_steps):
        estimator.train(train_input_fn, steps=n_steps)
        eval_results = estimator.evaluate(eval_input_fn)
        exporter.export(estimator, params['module_dir'], estimator.latest_checkpoint())
        print('model evaluation:', eval_results)

    return True

if __name__ == "__main__":
  tf.compat.v1.app.run()
