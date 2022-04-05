import scipy
import numpy as np
from sklearn.metrics import mutual_info_score

def find_group_stat(signals_a,signals_b,stat_function,noutput=1):
    ndim_a = signals_a.shape[1]
    ndim_b = signals_b.shape[1]
    if noutput>1:
        stats = [np.zeros([ndim_a,ndim_b]) for _ in range(noutput)]
    else:
        stats = np.zeros([ndim_a,ndim_b])
    for i in range(ndim_a):
        for j in range(ndim_b):
            x = signals_a[:,i]
            y = signals_b[:,j]
            result = stat_function(x,y)
            if noutput>1:
                for k in range(noutput):
                    stats[k][i,j] = result[k]
            else:
                stats[i,j] = result
    return stats

def find_correlation(signals_a,signals_b):
    correlation,stats = find_group_stat(signals_a,signals_b,scipy.stats.pearsonr,noutput=2)
    return correlation,stats

def find_l2_distance(signals_a,signals_b):
    l2d = lambda x,y:np.sqrt(np.sum(np.square(x-y)))
    return find_group_stat(signals_a,signals_b,l2d)

def find_l2_distance_modified(signals_a,signals_b):
    l2d = lambda x,y:np.min(np.sqrt(np.sum(np.square(x-y))),np.sqrt(np.sum(np.square(x+y))))
    return find_group_stat(signals_a,signals_b,l2d)

def find_l1_distance(signals_a,signals_b):
    l1d = lambda x,y:np.sum(np.abs(x-y))
    return find_group_stat(signals_a,signals_b,l1d)

def find_l1_distance_modified(signals_a,signals_b):
    l1d = lambda x,y:np.min(np.sum(np.abs(x-y)),np.sum(np.abs(x+y)))
    return find_group_stat(signals_a,signals_b,l1d)

def find_mutual(signals_a,signals_b):
    def get_mutual_info(x,y,bins = np.linspace(-1,1,1000)):
        pmfx,_ = np.histogram(x,bins)
        pmfy,_ = np.histogram(y,bins)
        return mutual_info_score(pmfx,pmfy)
    find_group_stat(signals_a,signals_b,get_mutual_info)

def custom_correlation(signalsa,signalsb,std = 1):
    nsiga = signalsa.shape[1]
    nsigb = signalsb.shape[1]
    corr = np.zeros([nsiga,nsigb])
    stats = np.zeros([nsiga,nsigb])
    for i in range(nsiga):
        for j in range(nsigb):
            siga = signalsa[:,i]
            sigb = signalsb[:,i]
            include = np.logical_or(np.abs(siga)>std*np.std(siga),np.abs(sigb)>std*np.std(sigb))
            siga = siga[include]
            sigb = sigb[include]
            corr[i,j],stats[i,j] = scipy.stats.pearsonr(siga,sigb)
    return corr,stats
