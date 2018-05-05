import scipy.io as sio
import numpy as np
import os

def get_train_test(mat_dat):
    train,test,mapping = mat_dat['dataset'][0][0]

    return train,test,mapping

def get_mapping(mat_dat):
    _,_,mapping = get_train_test(mat_dat)

    return mapping

def get_im_lab_writ(mat_dat,squeeze=True):
    train,test,mapping = get_train_test(mat_dat)

    x_train,y_train,w_train = train[0][0]
    x_test,y_test,w_test = test[0][0]

    if squeeze:
        y_train=np.squeeze(y_train)
        y_test=np.squeeze(y_test)
        w_train = np.squeeze(w_train)
        w_test = np.squeeze(w_test)

    return (x_train,y_train,w_train),(x_test,y_test,w_test)

def load_mnist(data_dir='~/data/emnist'):
    mat_fp = os.path.join(data_dir,'matlab','emnist-mnist.mat')
    dat = sio.loadmat(mat_fp)

    return get_im_lab_writ(dat)

def load_digits(data_dir='~/data/emnist'):
    mat_fp = os.path.join(data_dir,'matlab','emnist-digits.mat')
    dat = sio.loadmat(mat_fp)

    return get_im_lab_writ(dat)

def load_letters(data_dir='~/data/emnist'):
    mat_fp = os.path.join(data_dir,'matlab','emnist-letters.mat')
    dat = sio.loadmat(mat_fp)

    return get_im_lab_writ(dat)

def load_byclass(data_dir='~/data/emnist'):
    mat_fp = os.path.join(data_dir,'matlab','emnist-byclass.mat')
    dat = sio.loadmat(mat_fp)

    return get_im_lab_writ(dat)

def load_balanced():
    pass

