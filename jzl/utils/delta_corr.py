import scipy.stats as stats

def compare_iter(y_pred,y_true,start=0):
    if len(y_pred) - len(y_true) is not 0:
        raise ValueError('compared vectors are not same length')
    idx = start
    while idx < len(y_pred):
        yield (y_pred[idx],y_true[idx])
        idx += 1

def delta_corr(y_pred,y_true):

    pearson_delta = []
    for p,t in compare_iter(y_pred,y_true):
        pearson_delta.append(stats.pearsonr(p,t))

    return pearson_delta
