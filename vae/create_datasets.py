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

load_funcs=dict(mnist=ld.load_mnist, fmnist=ld.load_fmnist, cifar10=ld.load_cifar10)

def build_input_fns(data_type,batch_size,label):
    """Builds an iterator switching between train and heldout data."""

    print('loading %s dataset'%data_type)

    load_func = load_funcs[data_type]
    # these guys are already flattened
    x_train, y_train, x_test,y_test = load_func()
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
    #train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    #test_dataset  = tf.data.Dataset.from_tensor_slices((x_test,y_test))

    def train_input_fn():
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
        trainset = train_dataset.shuffle(max(train_sample_size,10000)).repeat().batch(batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(trainset)
        return iterator.get_next()

    def eval_input_fn():
        test_dataset  = tf.data.Dataset.from_tensor_slices((x_test,y_test))
        testset = test_dataset.shuffle(max(test_sample_size,10000)).repeat().batch(batch_size)
        return tf.compat.v1.data.make_one_shot_iterator(testset).get_next()

    return train_input_fn, eval_input_fn
