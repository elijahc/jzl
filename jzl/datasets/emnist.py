import scipy.io as sio
import numpy as np
import os
from ..utils.data_utils import get_file

def get_train_test(mat_dat):
    train,test,mapping = mat_dat['dataset'][0][0]

    return train,test,mapping

def get_mapping(mat_dat):
    _,_,mapping = get_train_test(mat_dat)

    return mapping

def get_im_lab_writ(mat_fp,squeeze=True):
    if not os.path.exists(mat_fp):
        get_file(fname='emnist.zip',
            dset_name='emnist',
            origin='http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip',
            extract=True,
            archive_format='zip')

    mat_dat = sio.loadmat(mat_fp)
    train,test,mapping = get_train_test(mat_dat)

    x_train,y_train,w_train = train[0][0]
    x_test,y_test,w_test = test[0][0]

    if squeeze:
        y_train=np.squeeze(y_train)
        y_test=np.squeeze(y_test)
        w_train = np.squeeze(w_train)
        w_test = np.squeeze(w_test)

    return (x_train,y_train,w_train),(x_test,y_test,w_test)

def load_mnist(data_dir=None):
    if data_dir is None:
        data_dir=os.path.expanduser('~/.datasets/emnist')

    mat_fp = os.path.join(data_dir,'matlab','emnist-mnist.mat')

    return get_im_lab_writ(mat_fp)

def load_digits(data_dir=None):
    if data_dir is None:
        data_dir=os.path.expanduser('~/.datasets/emnist')

    mat_fp = os.path.join(data_dir,'matlab','emnist-digits.mat')

    return get_im_lab_writ(mat_fp)

def load_letters(data_dir=None):
    if data_dir is None:
        data_dir=os.path.expanduser('~/.datasets/emnist')

    mat_fp = os.path.join(data_dir,'matlab','emnist-letters.mat')

    return get_im_lab_writ(mat_fp)

def load_byclass(data_dir=None):
    if data_dir is None:
        data_dir=os.path.expanduser('~/.datasets/emnist')

    mat_fp = os.path.join(data_dir,'matlab','emnist-byclass.mat')

    return get_im_lab_writ(mat_fp)

def load_balanced():
    if data_dir is None:
        data_dir=os.path.expanduser('~/.datasets/emnist')

    mat_fp = os.path.join(data_dir,'matlab','emnist-balanced.mat')

    return get_im_lab_writ(mat_fp)
