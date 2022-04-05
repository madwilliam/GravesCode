import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import butter, sosfilt, sosfreqz,filtfilt
import peakutils
import matplotlib.pyplot as plt

def baseline_als(y, lam=1000, p=0.001, niter=10):
  L = len(y)
  D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
  return z

def baseline_als_optimized(y, lam=1000, p=0.001, niter=10):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    D = lam * D.dot(D.transpose()) # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w) # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def apply_function_to_signals(function,signals,*arg,**kwargs):
    output = np.zeros(signals.shape)
    for i in range(signals.shape[1]):
        output[:,i] = function(signals[:,i],*arg,**kwargs)
    return output

def gaussian_filter_signal(signals,sigma = 50):
    return apply_function_to_signals(gaussian_filter1d,signals,sigma = sigma)

def normalize_range(signal):
    min = np.min(signal)
    max = np.max(signal)
    norm_factor = max - min
    return (signal-min)/norm_factor

def standardize(signal):
    return (signal-np.mean(signal))/np.std(signal)

def custom_norm(signals,norm_function = standardize):
    return apply_function_to_signals(norm_function,signals)

def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter_sos(data, lowcut, highcut, fs, order=5):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def differentiate(signal):
    return signal[:-1]-signal[1:]

def calculate_power_of_signal(signals):
    rectified = np.abs(signals)
    power = gaussian_filter_signal(rectified)
    return power

def find_baselines(signals,lam=1000, p=0.00001, niter=10):
    return apply_function_to_signals(baseline_als_optimized,signals,lam,p,niter)

def subtract_baseline_from_signals(signals, lam=1000, p=0.00001, niter=10):
    baselines = apply_function_to_signals(baseline_als_optimized,signals,lam,p,niter)
    return signals - baselines

def lowpass_fiter_signals(signals,cutoff,fs,order = 5):
    return apply_function_to_signals(butter_lowpass_filter,signals,cutoff,fs,order = order)

def find_baseline_peakutil(signals,deg=3, max_it=100, tol=1e-3):
    return apply_function_to_signals(peakutils.baseline,signals,deg, max_it, tol)

def plot_spectrum(signal,fs = 200):
    signal = np.random.rand(301) - 0.5
    ps = np.abs(np.fft.fft(signal))**2
    time_step = 1 / fs
    freqs = np.fft.fftfreq(signal.size, time_step)
    idx = np.argsort(freqs)
    plt.plot(freqs[idx], ps[idx])
    plt.xlim([0,fs/2])

def find_moving_average(signals,N = 50,window = lambda N: np.ones(N)/N):
    return apply_function_to_signals(np.convolve,signals, window(N), mode='same')

def find_mode(signal,nbins = 50):
    min = signal.min()
    max = signal.max()
    bins = np.linspace(min,max,nbins)
    count,val = np.histogram(signal,bins)
    return val[np.argmax(count)]

def custume_moving_average(signal,N = 100,nbins = 50):
    output = np.zeros(signal.shape)
    nsamples = len(signal)
    N = N-N%2
    npad = int(N/2)
    signal = np.pad(signal,[npad,npad],mode = 'symmetric')
    for si in range(nsamples):
        signal_in_window = signal[si:si+N]
        output[si] = find_mode(signal_in_window,nbins)
    return output

def find_custome_moving_average(signals,N=100,nbins = 50):
    return apply_function_to_signals(custume_moving_average,signals, N,nbins)
