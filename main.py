#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Thu Jul 11 09:32:01 2019
main.py 
run this script to  train your model
@author: nessa
'''

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

# standard packages
from absl import flags
import numpy as np
import functools
import os
import pickle as pkl

# tensorflow packages
import tensorflow as tf
import tensorflow_hub as hub

import vae.create_datasets as crd
from  vae.model import model_fn

flags.DEFINE_string('model_dir', default=os.path.join(os.path.abspath('./'),'model'), help='directory for storing the model (absolute path)')
flags.DEFINE_enum('data_set','mnist',['fmnist','cifar10','celeba','mnist','sn'], help='the tensorflow-dataset to load')
flags.DEFINE_string('module_dir', default=os.path.join(os.path.abspath('./'),'modules'), help='directory to which to export the modules (absolute path)')
flags.DEFINE_string('data_dir', default=os.path.join(os.path.abspath('./'),'data'), help='directory to store the data')

flags.DEFINE_float('learning_rate', default=1e-3, help='learning rate')    
flags.DEFINE_integer('batch_size',default=64, help='batch size')
flags.DEFINE_integer('max_steps', default=20000, help='training steps')    
flags.DEFINE_integer('n_steps', default=500, help='number of training steps after which to perform the evaluation')

flags.DEFINE_integer('latent_size',default=8, help='dimensionality of latent space')
flags.DEFINE_string('activation', default='leaky_relu', help='activation function')
flags.DEFINE_integer('n_samples', default=16, help='number of samples for encoding')
flags.DEFINE_enum('network_type', 'fully_connected', ['fully_connected','conv'], help='which type of network to use, currently supported: fully_conneted and conv')
flags.DEFINE_integer('n_filt',default=64,help='number of filters to use in the first convolutional layer')
flags.DEFINE_boolean('bias', default=False, help='whether to use a bias in the convolutions')
flags.DEFINE_boolean('AE', default=False, help='whether to run an AutoEncoder instead of a Variational AutoEncoder')
flags.DEFINE_boolean('add_noise', default=False, help='whether to add noise to the data before training')

flags.DEFINE_enum('likelihood','Gauss',['Gauss','Bernoulli'], help='form of likelihood')
flags.DEFINE_float('sigma', default=0.1, help='noise scale used in the Gaussian likelihood')
flags.DEFINE_integer('class_label', default=-1, help='number of specific class to train on. -1 for all classes')

FLAGS = flags.FLAGS

DATA_SHAPES = dict(mnist=[28,28,1],fmnist=[28,28,1],cifar10=[32,32,3],celeba=[64,64,3],sn=[256,1])

def main(argv):
    del argv

    params = FLAGS.flag_values_dict()
    DATA_SHAPE = DATA_SHAPES[FLAGS.data_set]

    params['activation']  = getattr(tf.nn, params['activation'])

    if len(DATA_SHAPE)>2:
        params['width']       = DATA_SHAPE[0]
        params['height']      = DATA_SHAPE[1]
        params['n_channels']  = DATA_SHAPE[2]
    else:
        params['length']      = DATA_SHAPE[0]
        params['n_channels']  = DATA_SHAPE[1]
    
    params['data_shape'] = DATA_SHAPE
    flatten = True

    params['output_size'] = np.prod(DATA_SHAPE)
    params['full_size']   = [None,params['output_size']] 

    if params['network_type']=='conv':
        flatten = False
        params['output_size'] = DATA_SHAPE
        params['full_size']   = [None,params['width'],params['height'],params['n_channels']]

    
    params['label']       = os.path.join('%s'%params['data_set'], '%s'%params['likelihood'], 'class%d'%params['class_label'], 'latent_size%d'%params['latent_size'],'net_type_%s'%params['network_type'])
    if params['AE']:
        params['label']+='AE'

    params['model_dir']   = os.path.join(params['model_dir'], params['label'])
    params['module_dir']  = os.path.join(params['module_dir'], params['label'])
    
    for dd in ['model_dir', 'module_dir', 'data_dir']:
        if not os.path.isdir(params[dd]):
            os.makedirs(params[dd], exist_ok=True)

    if not os.path.isdir('./params'):
        os.makedirs('./params')
    if params['AE']:
        pkl.dump(params, open('./params/params_%s_%s_%d_%d_%s-AE.pkl'%(params['data_set'],params['likelihood'],params['class_label'],params['latent_size'],params['network_type']),'wb'))
    else:
        pkl.dump(params, open('./params/params_%s_%s_%d_%d_%s.pkl'%(params['data_set'],params['likelihood'],params['class_label'],params['latent_size'],params['network_type']),'wb'))
 
    if params['data_set']=='celeba':
        input_fns      = crd.build_input_fn_celeba(params)
        train_input_fn = input_fns['train']
        eval_input_fn  = input_fns['validation']
    else:
        train_input_fn, eval_input_fn = crd.build_input_fns(params,label=FLAGS.class_label,flatten=flatten)

    estimator = tf.estimator.Estimator(model_fn, params=params, config=tf.estimator.RunConfig(model_dir=params['model_dir']),)
    c = tf.placeholder(tf.float32,params['full_size'])
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
