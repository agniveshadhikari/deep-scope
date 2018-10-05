# File formats
import csv
from scipy.io import wavfile
import numpy as np
# OS level ops
from glob import iglob
import os
# Local modules
from pcg.specrnn import spectrogram, kerasmodels
from pcg.datasets import PhysionetCinC as PCC
# Misc
from pprint import pprint
# Preprocessing
from keras.preprocessing.sequence import pad_sequences

# TODO implement command line args for this file. Num-epochs, batch size, dataset,
#       num_samples to be used for training so on and so forth


# TODO move the fetching dataset portion to the datasets.py file
#       Every dataset class shall have a get_dataset function that should return the dataset in (x, y) format directly


# Storing the spectrograms
specgrams = list()
for i in range(1, 401):
    wavfilepath = os.path.join(PCC.base_path, 'training-a', 'a'+str(i).zfill(4)+'.wav')
    fs, data = wavfile.read(wavfilepath)
    f, t, s = spectrogram.get(data, fs=fs)
    s = pad_sequences(s, 165)
    specgrams.append(s.T)

specgrams = np.array(specgrams)



# Loading the labels
labels = [
            1 if label=='1' else 0
            for filename, label
            in csv.reader(open(os.path.join(PCC.base_path,
                                            'training-a',
                                            'REFERENCE.csv')))
         ][:400]

print('1:', labels.count(1), '\t', '0:', labels.count(0))
labels = np.array(labels)

# Class count equalization, and separation of train and validation data
sorted_indices = np.argsort(labels)
composition = np.hstack([sorted_indices[0:100], sorted_indices[0:100], sorted_indices[200:400]])
labels = labels[composition]
specgrams = specgrams[composition]

v_composition = sorted_indices[100:200]
v_specgrams = specgrams[v_composition]
v_labels = labels[v_composition]

# Sanity Check
print(np.unique(labels, return_counts=True))


# Training
kerasmodels.model.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

kerasmodels.model.fit(specgrams, labels, batch_size=400, epochs=500, validation_data=(v_specgrams, v_labels))

preds = kerasmodels.model.predict(specgrams)

for i in range(400):
    print(preds[i], '\t', labels[i], '\t', np.round(preds[i])==labels[i])