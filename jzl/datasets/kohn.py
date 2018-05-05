import os
import numpy
import scipy.io as sio

from ..utils.data_utils import get_file

def _load_mat(mat_fp):
        if not os.path.exists(mat_fp):
            # Download dataset
            url = 'https://www.dropbox.com/s/nk3u1a58ooyq6ik/kohn.tar.gz?dl=1'
            get_file(fname='kohn_v1.tar.gz',
                    dset_name='kohn_v1',
                    origin=url,
                    extract=True,
                    archive_format='tar')

        mat_dat = sio.loadmat(mat_fp)

        return mat_dat

def load_images(dat_id):
    if dat_id in ['01','02','03','04','05','06','07','08','09','10']:

        mat_fp = os.path.join(os.path.expanduser('~'),'.datasets','kohn_v1','mat_files',dat_id+'.mat')
    else:
        raise 'Bad dat_id must be like "02"'

    dat = _load_mat(mat_fp)

    return dat['images']

def load_raw(dat_id):
    if dat_id in ['01','02','03','04','05','06','07','08','09','10']:

        mat_fp = os.path.join(os.path.expanduser('~'),'.datasets','kohn_v1','mat_files',dat_id+'.mat')
    else:
        raise 'Bad dat_id must be like "02"'

    dat = _load_mat(mat_fp)

    return dat
