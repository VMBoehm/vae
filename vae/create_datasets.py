#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:56:28 2019
retrieve_data.py
(down)load datasets and create tensorflow dataset
@author: nessa
"""

import vae.load_data as ld
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds
from functools import partial

load_funcs=dict(mnist=ld.load_mnist, fmnist=ld.load_fmnist, cifar10=ld.load_cifar10)

def build_input_fns(data_dir,data_type,batch_size,label,flatten):
    """Builds an iterator switching between train and heldout data."""

    print('loading %s dataset'%data_type)

    load_func = partial(load_funcs[data_type])
    # these guys are already flattened
    x_train, y_train, x_test,y_test = load_func(data_dir,flatten)
    num_classes = len(np.unique(y_train))

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

    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    def train_input_fn():
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        trainset = train_dataset.shuffle(max(train_sample_size,10000)).repeat().batch(batch_size,drop_remainder=True)
        iterator = tf.compat.v1.data.make_one_shot_iterator(trainset)
        return iterator.get_next()

    def eval_input_fn():
        test_dataset  = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        testset = test_dataset.shuffle(max(test_sample_size,10000)).batch(batch_size,drop_remainder=True)
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
