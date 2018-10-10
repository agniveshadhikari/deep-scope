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
from pcg.metrics import ConfusionMatrix
# Misc
from pprint import pprint
# Preprocessing
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle

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
    s_padded = pad_sequences(s, 165, dtype='float32')
    specgrams.append(s_padded.T)

specgrams = np.array(specgrams)



# Loading the labels
labels = np.array([
            1 if label=='1' else 0
            for filename, label
            in csv.reader(open(os.path.join(PCC.base_path,
                                            'training-a',
                                            'REFERENCE.csv')))
         ][:400])

# Class count equalization, and separation of train and test data. 
# TODO have better randomized train test splits. sklearn.model_selection.StratifiedKFold
sorted_indices = np.argsort(labels)
t_composition = np.hstack([sorted_indices[0:100], sorted_indices[0:100], sorted_indices[200:400]])
# t_composition = np.hstack([sorted_indices[0:100], sorted_indices[200:300]])
t_labels = labels[t_composition]
t_specgrams = specgrams[t_composition]

v_composition = sorted_indices[100:160]
# v_composition = np.hstack([sorted_indices[100:200], sorted_indices[300:400]])
v_specgrams = specgrams[v_composition]
v_labels = labels[v_composition]

# Shuffle the train set
t_specgrams, t_labels = shuffle(t_specgrams, t_labels)

# Sanity Check
print(np.unique(labels, return_counts=True))
print(np.unique(t_labels, return_counts=True))
print(np.unique(v_labels, return_counts=True))

# Training
kerasmodels.model.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

kerasmodels.model.fit(t_specgrams, t_labels, batch_size=400, epochs=5, validation_data=(v_specgrams, v_labels))

t_preds = np.round(kerasmodels.model.predict(t_specgrams))

v_preds = np.round(kerasmodels.model.predict(v_specgrams))

# Visualization of results
# for i in range(400):
#     print(preds[i], '\t', labels[i], '\t', np.round(preds[i])==labels[i])

ConfusionMatrix(t_labels, t_preds, ['Normal', 'Abnormal']).plot('Test Confusion Matrix')
ConfusionMatrix(v_labels, v_preds, ['Normal', 'Abnormal']).plot('Validation Confusion Matrix')