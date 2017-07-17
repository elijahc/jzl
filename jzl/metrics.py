import numpy as np
from scipy.stats import circvar

def si(activity):
    # activity [nsamples, ncells]
    nb_imgs = activity.shape[0]
    nb_neurons = activity.shape[1]
    si = []
    for n in np.arange(nb_neurons):
        sfr = [ np.power(fr,2)/nb_imgs for fr in activity[:,n]]
        sfr = np.array(sfr)
        num = np.power((activity[:,n]/nb_imgs).sum(),2)
        sp = (1 - num/sfr.sum())*(1-(1/nb_imgs))**-1
        si.extend([ sp ])
    return np.array(si)

def gen_y_fake(y, sem_y):
    loc = np.zeros_like(y)
    z = np.random.normal(loc,sem_y)
    return (y + z)

def pairwise_pcc(y,y_pred):
    # Expects data in shape [nsamples, ncells]

    ncells = y.shape[1]
    ppcc = [ np.corrcoef(y_pred[:,i],y[:,i]) for i in np.arange(ncells)]
    return np.nan_to_num(np.array(ppcc)[:,1,0])

def ppcc_max(activity_mean, activity_sem):
    # Expects shape [nsamples, ncells]

    y_fake = gen_y_fake(activity_mean, activity_sem)
    return pairwise_pcc(y_fake, activity_mean)

if __name__ == '__main__':
    import scipy.io as sio
    save_dir = '/home/elijahc/Dropbox/Kohn_Monkey_Data/v1_predictor_paper/data'

    mat_file = '../data/02_stats.mat'
    print('loading mat data...', mat_file)
    activity_contents = sio.loadmat(mat_file)
    activity = activity_contents['resp_mean'].swapaxes(0,1)
    activity_sem = activity_contents['resp_sem'].swapaxes(0,1)

    # Natural Images
    idxs = np.arange(540)[::2]

    grating_idxs = np.arange(668,732)
    grating_activity = activity[grating_idxs].reshape(16,4,-1).swapaxes(0,1)

    L_ori = 1 - circvar(grating_activity, axis=1)

    ppcc_max = ppcc_max(activity[idxs], activity_sem[idxs])

    si = si(activity[idxs])

    filename = '02_metrics.mat'
    print('writing ','/',filename)
    sio.savemat(save_dir+'/02_metrics.mat', dict( L_ori=L_ori, ppcc_max=ppcc_max, si=si ))
