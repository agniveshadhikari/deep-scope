from librosa.feature import mfcc
from librosa.display import specshow
import matplotlib.pyplot as plt

class MFCC():
    def __init__(self, y, sampling_rate=2000, n_mfcc=20, isBatch=True):
        self.isbatch = False # Figure out how to determine
        self.y = y
        self.sampling_rate = sampling_rate
        self.n_mfcc = n_mfcc

    def get(self):
        if not self.isbatch:
            return mfcc(y=self.y,
                        sr=self.sampling_rate,
                        n_mfcc=self.n_mfcc)

        else: # is batch
            return [mfcc(y=y,
                         sr=self.sampling_rate,
                         n_mfcc=self.n_mfcc)
                    for y in self.y]

    def plot(self):
        if not self.isbatch:
            mfccs = mfcc(y=self.y,
                         sr=self.sampling_rate,
                         n_mfcc=self.n_mfcc)

        else: # is batch
            mfccs = mfcc(y=self.y[0],
                         sr=self.sampling_rate,
                         n_mfcc=self.n_mfcc)

        plt.figure(figsize=(10, 4))
        specshow(mfccs, x_axis='time')
        plt.show()

    def get_shape(self):
        pass