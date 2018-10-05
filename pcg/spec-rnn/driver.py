from scipy.io import wavfile
import spectrogram

for i in range(1, 10):

    fs, data = wavfile.read('/home/agnivesh/Desktop/fyp/pcg/physionet-cinc/Dataset/training/training-a/a000{}.wav'.format(str(i)))
    f, t, s = spectrogram.get(data, fs)
    spectrogram.plot(data, fs)
