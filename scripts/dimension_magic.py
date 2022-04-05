from sklearn.decomposition import PCA
from FileIO import FileIO
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.decomposition import FastICA
from statsmodels.stats.diagnostic import acorr_ljungbox
from signal_utilities import custom_norm
import mne

def get_mutual_information(x,y):
    bins = np.linspace(-128.5,128,128*2+1)
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi

def time_delay_embed(x,ndelay=10,tau=1):
    nsample = len(x)
    final_length = nsample-ndelay*tau+1
    embeded_signal = []
    for i in range(ndelay):
        embeded_signal.append(x[i*tau:i*tau+final_length])
    return np.vstack(embeded_signal)

def find_time_delay_MI(x,ndelay = 50,tau = 1):
    embedding = time_delay_embed(x,ndelay=ndelay,tau=tau)
    mis = []
    for i in range(ndelay-1):
        mis.append(get_mutual_information(embedding[0],embedding[i+1]))
    return(mis)

def load_emg_data():
    io = FileIO()
    exps = io.load_experiment('s1')
    emg = exps['S1_E1_A1']['emg'].astype(int)
    emg[:,:8] = emg[:,:8] - np.mean(emg[:,:8],axis=1).reshape(-1,1)
    emg[:,8:] = emg[:,8:] - np.mean(emg[:,8:],axis=1).reshape(-1,1)
    return emg

def get_time_delayed_embedding(signals):
    phase_space_reconstruction = []
    for i in range(16):
        phase_space_reconstruction.append(time_delay_embed(signals[:,i],ndelay = 7,tau = 400))
    phase_space_reconstruction = np.vstack(phase_space_reconstruction)
    print(f'The reconstructed phase space has {phase_space_reconstruction.shape[0]} dimensions with {phase_space_reconstruction.shape[1]} time samples each')
    return phase_space_reconstruction

def pca_analysis(signals):
    pca = PCA(svd_solver='auto')
    pca.fit(signals.T)
    # plt.title('Variance explained with first n Principle Components')
    # plt.plot(np.cumsum(pca.explained_variance_ratio_))
    ndim = np.where(np.cumsum(pca.explained_variance_ratio_)>0.95)[0][0]
    print(f'first {ndim} principle components catpures 95 percent of all variances')
    pc = pca.transform(signals.T)
    reconstruction = pc[:,:ndim]
    return reconstruction

def calculate_ljungbox(signals):
    sig_len = 10000
    p_val = []
    stats = []
    for i in range(signals.shape[1]):
        sig = signals[:sig_len,i]
        res = acorr_ljungbox(sig, lags=[1], return_df=True)
        p_val.append(float(res['lb_pvalue']))
        stats.append(float(res['lb_stat']))
    stats = np.array(stats)
    return stats

def get_time_independent_signal(signals,threshold = 800):
    stats = calculate_ljungbox(signals)
    time_dependent = stats>threshold
    print(f'There are {sum(time_dependent)} sources with time dependent signals')
    time_dependent_sources = signals[:,time_dependent]
    return time_dependent_sources

def fast_ica_analysis(signals):
    transformer = FastICA(random_state=0)
    signals_transformed = transformer.fit_transform(signals)
    return signals_transformed

def info_max_ica(signals):
    _, S_pred, _ = mne.preprocessing.infomax(signals)
    return S_pred.T

def amica(signals):
    ...

def get_emg_ica(ica_function = fast_ica_analysis):
    emg = load_emg_data()
    phase_space_reconstruction = get_time_delayed_embedding(emg)
    reconstruction = pca_analysis(phase_space_reconstruction)
    EMG_transformed = ica_function(reconstruction)
    time_dependent_sources = get_time_independent_signal(EMG_transformed)
    return time_dependent_sources

def get_glove_data():
    io = FileIO()
    exps = io.load_experiment('s1')
    gloves = exps['S1_E1_A1']['glove']
    return gloves

def get_glove_ica():
    gloves = get_glove_data()
    transformer = FastICA(random_state=1)
    gloves_transformed = transformer.fit_transform(gloves)
    return gloves_transformed

def get_emg_and_glove_components(ica_function = fast_ica_analysis,norm_function = custom_norm):
    time_dependent_sources = get_emg_ica(ica_function = ica_function)
    gloves_transformed = get_glove_ica()
    gloves_transformed = gloves_transformed[:len(time_dependent_sources),:]
    gloves_transformed=norm_function(gloves_transformed)
    time_dependent_sources = norm_function(time_dependent_sources)
    return time_dependent_sources,gloves_transformed
