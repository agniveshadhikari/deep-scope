import scipy.signal as signal
import matplotlib.pyplot as plt

def get(data, fs):
    # TODO Check the doc page for signal.spectrogram. Below, there are
    # a couple of different methods to find the PSD for a waveform.
    # One task is to compare how those match up to regular spectrograms.

    # TODO The frequencis are consistent. But the interval is 7.something
    # Find if it is actually fixed to this arbitrary 7.xx and why

    # TODO One hyperparameter is the spectrogram resolution in the time domain.
    # Find the lowest res which can still give the base accuracy.
    # Less samples => less compute
    f, t, s = signal.spectrogram(x=data, fs=fs)
    return f, t, s

def plot(data, fs):
    f, t, s = signal.spectrogram(x=data, fs=fs)
    plt.pcolormesh(t, f, s)
    plt.ylim(ymax=400)
    plt.xlim(xmax=10)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()