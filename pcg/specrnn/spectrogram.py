import scipy.signal as signal
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# TODO Wrap the functions into a class, with instance vars fs and nperseg
# TODO Implement the scaling arg in all funcs

default_nperseg = 100

def get(data, fs, nperseg=None, noverlap=None, scaled=False):
    # TODO Check the doc page for signal.spectrogram. Below, there are
    # a couple of different methods to find the PSD for a waveform.
    # Compare how those match up to regular spectrograms.
    # TODO One hyperparameter is the spectrogram resolution in the time domain.
    # Find the lowest res which can still give the base accuracy.
    # TODO Another hyperarameter is the frequency domain resolution.
    # Less samples => less compute
    # TODO Check if scaling is being done in the correct axis
    if nperseg is None:
        nperseg = default_nperseg
    f, t, s = signal.spectrogram(x=data, fs=fs, nperseg=nperseg, noverlap=noverlap)

    if scaled:
        s_scaled = StandardScaler().fit_transform(s.T).T
        s = s_scaled

    return f, t, s

def plot(data, fs, nperseg=None, noverlap=None):
    f, t, s = get(data, fs, nperseg, noverlap)
    plt.pcolormesh(t, f, s)
    plt.ylim(ymax=600)
    plt.xlim(xmax=10)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

    return f, t, s

def save(data, fs, path):
    # TODO Complete this. Right now it's not required 
    # as computation of spectrograms is not a bottleneck at all.
    f, t, s = get(data, fs)
    
def batch_get(data_iterable, fs):
    specgrams = list()
    
    for waveform in data_iterable:
        f, t, s = get(waveform, fs)
        specgrams.append(s)

    return specgrams


def _get_freq_resolution(fs, nperseg):
    pass

def _get_time_resolution(fs, noverlap):
    pass