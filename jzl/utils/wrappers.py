import scipy.io as sio
import numpy as np

class MatWrapper(object):

    def __init__(self,mat_file):
        self.mat_fp = mat_file

        self.data = None

class NeuroSurgMat(MatWrapper):

    def __init__(self, mat_file):
        self.mat_fp = mat_file
        self.data = None
        self._clfp = None
        self._cmacro_lfp = None

    @property
    def CLFP(self):
        # Lazy load CLFP files
        if self.data is None:
            self.data = sio.loadmat(self.mat_fp)
        if self._clfp is None:
            clfp = np.empty((3,self.data['CLFP_01'].shape[1]))
            for i in np.arange(3):
                clfp[i,:] = np.squeeze(self.data['CLFP_0'+str(i+1)])
            self._clfp = clfp
        return self._clfp

    @property
    def CMacro_LFP(self):
        if self.data is None:
            self.data = sio.loadmat(self.mat_fp)
        if self._cmacro_lfp is None:
            cmacro_lfp = np.empty((3,self.data['CMacro_LFP_01'].shape[1]))
            for i in np.arange(3):
                cmacro_lfp[i,:] = np.squeeze(self.data['CMacro_LFP_0'+str(i+1)])
            self._cmacro_lfp = cmacro_lfp
        return self._cmacro_lfp
