from librosa.feature import mfcc
from librosa.display import specshow
import matplotlib.pyplot as plt

class MFCC():
    def __init__(self, y, sampling_rate, nperseg, stride, n_mfcc):
        self.isbatch = True # Figure out how to determine
        self.y = y
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc
        self.nperseg = nperseg
        self.stride = stride

    def _mfcc(self, y):
        return mfcc(y=y,
                    sr=self.sampling_rate,
                    n_mfcc=self.n_mfcc,
                    n_fft=self.nperseg,
                    hop_length=self.stride)

    def get(self):
        if not self.isbatch:
            return self._mfcc(y=self.y.astype('float32'))

        else: # is batch
            mfccss = list()

            for y in self.y:
                mfccss.append(self._mfcc(y=y.astype('float32')))

            return mfccss

    def plot(self):
        # librosa is stupid. The melspectrogram call doesn't return the
        # x-axis(time) array, that pcolormesh needs (and therefore specshow needs)
        # specshow messes up the time axix labels therefore. Hence, please manually 
        # infer the timestamps
        mfccs = self.get()

        if self.isbatch:
            mfccs = mfccs[0]

        plt.figure(figsize=(10, 4))
        specshow(mfccs)
        plt.show()

    def get_shape(self):
        pass