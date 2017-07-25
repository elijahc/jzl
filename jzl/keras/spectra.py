import numpy as np
from keras.engine.topology import Layer
from jzl.utils.spectra import rolling_window
import mtspec.multitaper as mtm

class RollingWindow(Layer):

    def __init__(self,window_size,
                 window_overlap=0,
                 **kwargs):
        super(RollingWindow,self).__init__(**kwargs)
        self.win = window_size
        self.overlap = window_overlap

    def build(self, input_shape):
        self.built = True

    def compute_output_shape(self, input_shape):
        # Output shape (batch, time, window_size)
        output_shape = (input_shape[0], input_shape[1]//self.win, self.win)
        return output_shape

    def call(self, x):
        output_shape = compute_output_shape(x.shape)
        output = np.empty(shape=output_shape)
        for i in np.arange(output_shape[0]):
            output[i] = rolling_window(x[i],self.win)
        return output

class PowerSpectrum(Layer):

    def __init__(self,Fs,
                df,
                 **kwargs):

        super(PowerSpectrum, self).__init__(**kwargs)
        self.Fs = Fs
        self.df = df

    def build(self, input_shape):
        # Implement this
        win = input_shape[-1]
        tw = self.df*win/self.Fs/2
        L = int(2*tw)-1
        self.tapers,_,_ = mtm.dpss(win,tw,L)
        self.built = True

    def call(self,x):
        # Input shape: (None, t, window_size)
        output = x
        return output

    def compute_output_shape(self, input_shape):
        # Output shape (batch, time, t_signals, freq_bins)
        output_shape = (input_shape[0], input_shape[1], input_shape[2], self.win//2)
        return output_shape
