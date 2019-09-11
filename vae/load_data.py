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


def _get_datafolder_path():
    """
    returns data path
    """
    #where am I? return full path
    full_path = os.path.abspath('./')
    path = full_path +'/data'
    return path


def load_mnist(flatten=True):
    """
    load mnist dataset
    """

    dataset=_get_datafolder_path()+'/mnist/mnist.pkl.gz'

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
        x_valid, targets_valid = valid_set[0], valid_set[1]
        x_test,  targets_test  = test_set[0], test_set[1]
    else:
        x_train, targets_train = train_set[0].reshape((-1,28,28,1)), train_set[1]
        x_valid, targets_valid = valid_set[0].reshape((-1,28,28,1)), valid_set[1]
        x_test,  targets_test  = test_set[0].reshape((-1,28,28,1)), test_set[1]

        #omitting validation set for consistency
    return x_train, targets_train, x_test, targets_test


def load_fmnist():
    
    x_train, y_train = mnist_reader.load_mnist('./data/fashion', kind='train')
    x_test, y_test = mnist_reader.load_mnist('./data/fashion', kind='t10k')
    x_train = x_train/255.
    x_test  = x_test/255.
    return x_train, y_train, x_test, y_test


def reshape_cifar(x):
    x = x.reshape([-1, 3, 32, 32])
    x = x.transpose([0, 2, 3, 1])
    return x.reshape(-1,3*32*32)

def load_cifar10():
    """   
    load cifar10 dataset
    """

    dataset=_get_datafolder_path()+'/cifar10/cifar-10-python.tar.gz'

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

    train_x = reshape_cifar(train_x)
    test_x  = reshape_cifar(test_x)
    return train_x, train_y, test_x, test_y


def load_Gaussian_mnist(masking=0,mode=0,path=0):

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
        
    return  x_train, y_train, x_test, y_test, dict(covs=covs, means=means)


def load_Gaussian_data(filename, train_num, test_num):
    
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

    return  x_train, y_train, x_test, y_test, dict(covs=covs, means=means)  
