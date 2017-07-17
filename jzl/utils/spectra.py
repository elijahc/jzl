import numpy as np

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def mtspec(data,win,df=1):
    import mtspec.multitaper as mtm
    win = int(win)
    tw = int(df*win/2)
    L = int(2*tw)-1

    tapers,lamb,theta = mtm.dpss(win,tw,L)
    tapers = np.swapaxes(tapers,0,1)
    fft_split_data = []
    for i in np.arange(data.shape[0]):
        rep_data = np.matlib.repmat(data[i],L,1)
        split_data = np.multiply(rep_data,tapers)
        fft_data = np.fft.fft(split_data)[:,1:win//2]
        fft_split_data.append(fft_data)

    mtspec_data = np.array(fft_split_data).mean(axis=1)
    mtpspec = 10*np.log10(np.abs(mtspec_data.swapaxes(0,1))**2)
    return (mtpspec,tapers)
