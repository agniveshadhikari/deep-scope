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
from pcg.metrics import ConfusionMatrix, TrainingHistory
# Misc
from pprint import pprint
# Getting the dataset in order
from keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from mlutils.classbalancing import oversample

# TODO implement command line args for this file. Num-epochs, batch size, dataset,
#       num_samples to be used for training so on and so forth

# TODO Maybe move spectrogram padding to spectrogram.py


# Get the data for training-a
print('Reading Dataset...')
data = PCC.get_subset('training-a')

# Compute the spectrograms and add as a column to data
print('Computing Spectrograms...')
data['spectrogram'] = [pad_sequences(s, 900, 'float32').T
                       for s in spectrogram.batch_get(data['waveform'], data['fs'][0])]

# Make train test split
# TODO Use stratification? stratify=data['label'] should work in my understanding
#       But the default is to shuffle, so statistically it shouldn't be a problem
print('Splitting dataset into test and train sets...')
data_train, data_test = train_test_split(data, test_size=0.2)

# Sanity Check
print('Test Train Split Stats:\n')
print('Training size:', len(data_train))
print('Test size:   :', len(data_test))
print('Class Distribution Training:', np.unique(data_train['label'], return_counts=True))
print('Class Distribution Test    :', np.unique(data_test['label'], return_counts=True))

# Oversample train and test separately
print('Oversampling to balance classes...')
data_train = oversample(data_train, 'label')
data_test = oversample(data_test, 'label')

# Sanity Check
print('Test Train Split Stats after Oversampling:\n')
print('Training size:', len(data_train))
print('Test size:   :', len(data_test))
print('Class Distribution Training:', np.unique(data_train['label'], return_counts=True))
print('Class Distribution Test    :', np.unique(data_test['label'], return_counts=True))

# Training
print('Training the model...')
kerasmodels.model.compile(loss='binary_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

history = kerasmodels.model.fit(np.stack(data_train['spectrogram']), np.stack(data_train['label']),
                      batch_size=500, 
                      epochs=10,
                      validation_data=(np.stack(data_test['spectrogram']), np.stack(data_test['label'].values)))

# Training history visualization
TrainingHistory(history).plot()

# Confusion Matrix
print('Calculating the Confusion Matrices...')
train_preds = np.round(kerasmodels.model.predict(np.stack(data_train['spectrogram']))).astype('int32')
test_preds = np.round(kerasmodels.model.predict(np.stack(data_test['spectrogram']))).astype('int32')

ConfusionMatrix(data_train['label'].values, train_preds, ['Normal', 'Abnormal']).plot('Test Confusion Matrix')
ConfusionMatrix(data_test['label'].values, test_preds, ['Normal', 'Abnormal']).plot('Validation Confusion Matrix')