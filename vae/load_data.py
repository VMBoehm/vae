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

# functions to load different datasets (mnist, cifar10, random Gaussian data) 

import gzip, zipfile, tarfile
import os, shutil, re, string, urllib, fnmatch
import pickle as pkl
import numpy as np
import sys

try:
    try:
        sys.path.append('../fashion-mnist/utils/')
        import mnist_reader
    except:
        sys.path.append('/home/nessa/Documents/codes/fashion-mnist/utils/')
        import mnist_reader
except:
    print('did not import fashion mnist reader')
        
def _download_mnist(dataset):
    """
    download mnist dataset if not present
    """
    origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
    print('Downloading data from %s' %origin)
    urllib.request.urlretrieve(origin, dataset)


def _download_cifar10(dataset):
    """
    download cifar10 dataset if not present
    """
    origin = ('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    
    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin,dataset)

def _download_fmnist(dataset,subset,labels=False):

    if subset=='test':
        subset = 't10k'
    if labels:
        origin = ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/%s-labels-idx1-ubyte.gz'%subset)
    else:
        origin = ('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/%s-images-idx3-ubyte.gz'%subset)

    print('Downloading data from %s' % origin)
    urllib.request.urlretrieve(origin,dataset)


def _get_datafolder_path():
    """
    returns data path
    """
    #where am I? return full path
    full_path = os.path.abspath('./')
    path = full_path +'/data'
    return path


def load_mnist(data_dir,flatten=True,add_noise=False):
    """
    load mnist dataset
    """

    dataset=os.path.join(data_dir,'mnist/mnist.pkl.gz')

    if not os.path.isfile(dataset):
        datasetfolder = os.path.dirname(dataset)
        if not os.path.exists(datasetfolder):
            print('creating ', datasetfolder)
            os.makedirs(datasetfolder)
        _download_mnist(dataset)

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pkl.load(f, encoding='latin1')
    f.close()

    if flatten:
        x_train, targets_train = train_set[0], train_set[1]
        x_test,  targets_test  = test_set[0], test_set[1]
    else:
        x_train, targets_train = train_set[0].reshape((-1,28,28,1)), train_set[1]
        x_test,  targets_test  = test_set[0].reshape((-1,28,28,1)), test_set[1]

    if add_noise:
        x_train = add_white_noise(x_train)
        x_test  = add_white_noise(x_test)

    return x_train, targets_train, x_test, targets_test


def load_fmnist(data_dir,flatten=True, add_noise=False):
   
    data = {}
    for subset in ['train','test']:
        data[subset]={}
        for labels in [True,False]:
            if labels:
                dataset=os.path.join(data_dir,'fmnist/fmnist_%s_labels.gz'%subset)
            else:
                dataset=os.path.join(data_dir,'fmnist/fmnist_%s_images.gz'%subset)
            datasetfolder = os.path.dirname(dataset)
            if not os.path.isfile(dataset):
                if not os.path.exists(datasetfolder):
                    os.makedirs(datasetfolder)
                _download_fmnist(dataset,subset,labels)
            with gzip.open(dataset, 'rb') as path:
                if labels:
                    data[subset]['labels'] = np.frombuffer(path.read(), dtype=np.uint8,offset=8)
                else:
                    data[subset]['images'] = np.frombuffer(path.read(), dtype=np.uint8,offset=16)
                    
    x_train = data['train']['images'].reshape((-1,28*28))/255.
    x_test  = data['test']['images'].reshape((-1,28*28))/255.
    if not flatten:
        x_train = x_train.reshape((-1,28,28,1))
        x_test  = x_test.reshape((-1,28,28,1))

    y_train = data['train']['labels']
    y_test  = data['test']['labels']

    if add_noise:
        x_train = add_white_noise(x_train)
        x_test  = add_white_noise(x_test)

    return x_train, y_train, x_test, y_test


def reshape_cifar(x,flatten):
    x = x.reshape([-1, 3, 32, 32])
    x = x.transpose([0, 2, 3, 1])
    if flatten:
        x.reshape(-1,3*32*32)
    return x

def load_cifar10(data_dir,flatten=True, add_noise=False):
    """   
    load cifar10 dataset
    """

    dataset = os.path.join(data_dir,'cifar10/cifar-10-python.tar.gz')

    datasetfolder = os.path.dirname(dataset)
    if not os.path.isfile(dataset):
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_cifar10(dataset)
        with tarfile.open(dataset) as tar:
            tar.extractall(path=datasetfolder)
        
    for i in range(5):
        batchName = os.path.join(datasetfolder,'cifar-10-batches-py/data_batch_{0}'.format(i + 1))
        with open(batchName, 'rb') as f:
            d = pkl.load(f, encoding='latin1')
            data = d['data']/255.
            label= d['labels']
            try:
                train_x = np.vstack((train_x,data))
            except:
                train_x = data
            try:
                train_y = np.append(train_y,np.asarray(label))
            except:
                train_y = np.asarray(label)
                
    batchName = os.path.join(datasetfolder,'cifar-10-batches-py/test_batch')
    with open(batchName, 'rb') as f:
        d = pkl.load(f, encoding='latin1')
        data = d['data']/255.
        label= d['labels']
        test_x = data
        test_y = np.asarray(label)

    train_x = reshape_cifar(train_x,flatten)
    test_x  = reshape_cifar(test_x,flatten)

    if add_noise:
        x_train = add_white_noise(x_train)
        x_test  = add_white_noise(x_test)


    return train_x, train_y, test_x, test_y


def load_sn_lightcurves(data_dir,flatten=True, train_frac=0.8, add_noise=False):

    dataset     = os.path.join(data_dir,'lightcurves/salt2','salt2_spectra_downsampled_deredshifted.npy')
    
    wl, spectra = np.load(dataset, allow_pickle=True)
    spectra/=np.mean(spectra,axis=0)
    num         = len(spectra)
    len_train   = int(num*train_frac)
    x_train     = spectra[0:len_train]
    x_test      = spectra[len_train::]
    y_train     = np.empty((len_train))
    y_test      = np.empty((num-len_train))

    if add_noise:
        x_train = add_white_noise(x_train)
        x_test  = add_white_noise(x_test)

    return x_train, y_train, x_test, y_test


def load_Gaussian_mnist(masking=0,mode=0,path=0,add_noise=False):

    if 0 in [masking,mode,path]:
        filename='../data/Gaussian_mnist/ML_inpainted.pkl'
    else:
        if masking:
            label='masked'
        else:
            label='inpainted'
        filename='%s_%s.pkl'%(mode,label)
        filename=os.path.join(path,filename)

    data, covs, means = pkl.load(open(filename,'rb'))
    labels=[]
    for ii in range(len(data)):
        labels.append(np.ones((len(data[ii])))*ii)
    labels = np.asarray(labels)
    
    x_train = data[:,:5000,:].reshape(-1,data.shape[-1])
    y_train = labels[:,:5000].flatten()
    x_test  = data[:,5000:,:].reshape(-1,data.shape[-1])
    y_test  = labels[:,5000:].flatten()

    if add_noise:
        x_train = add_white_noise(x_train)
        x_test  = add_white_noise(x_test)

    return  x_train, y_train, x_test, y_test, dict(covs=covs, means=means)



def load_Gaussian_data(filename, train_num, test_num, add_noise=False):
    
    data, covs, means = pkl.load(open(filename+'_num%d_train.pkl'%train_num,'rb'))

    labels=[]
    for ii in range(len(data)):
        labels.append(np.ones((len(data[ii])))*ii)
    labels = np.asarray(labels)

    x_train = data.reshape(-1, data.shape[-1])
    y_train = labels.flatten()

    data,_,_ = pkl.load(open(filename+'_num%d_test.pkl'%test_num,'rb'))
     
    labels=[]
    for ii in range(len(data)):
        labels.append(np.ones((len(data[ii])))*ii)
    labels = np.asarray(labels)

    x_test = data.reshape(-1, data.shape[-1])
    y_test = labels.flatten()   

    if add_noise:
        x_train = add_white_noise(x_train)
        x_test  = add_white_noise(x_test)

    return  x_train, y_train, x_test, y_test, dict(covs=covs, means=means) 


def add_white_noise(x,level=0.01):
    #input data normalized?
    assert(np.max(x)<=1)
    assert(np.min(x)>=0)

    wn = np.random.randn(x.shape)*level

    return x+wn
