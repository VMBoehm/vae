#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:56:28 2019
retrieve_data.py
(down)load datasets and create tensorflow dataset
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

import vae.load_data as ld
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
from functools import partial

load_funcs=dict(mnist=ld.load_mnist, fmnist=ld.load_fmnist, cifar10=ld.load_cifar10, sn=ld.load_sn_lightcurves)

def add_noise(x,sigma=0.1):
    nn = tf.random.normal(tf.shape(x), dtype=tf.float32)
    x  = x+nn*sigma
    return x

def rotate(x,max_ang=10.):
    max_ang   = max_ang*np.pi/180.
    shape     = tf.shape(x)
    batchsize = shape[0]
    square    = tf.math.reduce_prod(shape[1:])
    onedim    = tf.cast(tf.math.sqrt(tf.cast(square,dtype=tf.float32)),dtype=tf.int32)
    rot_ang = tf.random.uniform([batchsize],minval=-max_ang, maxval=max_ang)
    x = tf.reshape(x, shape=[batchsize,onedim,onedim,1])
    x = tf.contrib.image.rotate(x,rot_ang)
    x = tf.reshape(x,shape)
    return x

def build_input_fns(params,label,flatten):
    """Builds an iterator switching between train and heldout data."""

    print('loading %s dataset'%params['data_set'])

    load_func                       = partial(load_funcs[params['data_set']])
    x_train, y_train, x_test,y_test = load_func(params['data_dir'],flatten)
    num_classes                     = len(np.unique(y_train))
    
    augment                         = params['augment']
    
    if label in np.arange(num_classes):
        index   = np.where(y_train==label)
        x_train = x_train[index]
        y_train = y_train[index]
        index   = np.where(y_test==label)
        x_test  = x_test[index]
        y_test  = y_test[index]
    elif label ==-1:
        pass
    else:
        raise ValueError('invalid class')
    
    train_sample_size = len(x_train)
    test_sample_size  = len(x_test)

    x_train  = x_train.astype(np.float32)
    shape    = [params['batch_size']]+[ii for ii in x_train.shape[1:]]
    x_test   = x_test.astype(np.float32)

    def train_input_fn():
        def mapping_function(x):
            def extract_images(inds):
                return x_train[inds]
            xx = tf.py_func(extract_images,[x],tf.float32)
            xx.set_shape(shape)
            return xx

        train_dataset  = tf.data.Dataset.range(train_sample_size)
        trainset       = train_dataset.shuffle(max(train_sample_size,10000)).repeat().batch(params['batch_size'],drop_remainder=True)
        trainset       = trainset.map(mapping_function)
        if augment=='noise':
            trainset   = trainset.map(add_noise)
        elif augment=='rotate':
            trainset   = trainset.map(rotate)
        else:
            pass
        iterator = tf.compat.v1.data.make_one_shot_iterator(trainset)
        return iterator.get_next()

    def eval_input_fn():
        def mapping_function(x):
            def extract_images(inds):
                return x_test[inds]
            xx = tf.py_func(extract_images,[x],tf.float32)
            xx.set_shape(shape)
            return xx

        test_dataset  = tf.data.Dataset.range(test_sample_size)
        testset       = test_dataset.shuffle(max(test_sample_size,10000)).repeat(2).batch(params['batch_size'],drop_remainder=True)
        testset       = testset.map(mapping_function)
        return tf.compat.v1.data.make_one_shot_iterator(testset).get_next()

    return train_input_fn, eval_input_fn


def build_input_fn_celeba(params):
    """
    Creates input functions for training, validation, and testing
    """

    def input_fn(is_training=False, tag='train', shuffle_buffer=10000):

        data_dir = os.path.join(params['data_dir'],'celeba/')

        if not os.path.isdir(data_dir):
            os.makedirs(data_dir)

        data = tfds.load("celeb_a", with_info=False, data_dir=data_dir)

        dset = data[tag]
        # Crop celeb image and resize
        dset = dset.map(lambda x: tf.cast(tf.image.resize_image_with_crop_or_pad(x['image'], 128, 128), tf.float32) / 256.)

        if is_training:
            dset = dset.repeat()
            dset = dset.map(lambda x: tf.image.random_flip_left_right(x),num_parallel_calls=2)
            dset = dset.shuffle(buffer_size=shuffle_buffer)

        dset = dset.batch(params['batch_size'],drop_remainder=True)
        dset = dset.map(lambda x: tf.image.resize_bilinear(x, (64, 64)),num_parallel_calls=2)
        dset = dset.prefetch(params['batch_size'])
        return dset

        return input_fn

    return {'train':partial(input_fn, tag='train', is_training=True),
            'validation': partial(input_fn, tag='validation'),
            'test': partial(input_fn, tag='test')}
