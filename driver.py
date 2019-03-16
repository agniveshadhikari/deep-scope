# File formats
import csv
from scipy.io import wavfile
from pprint import pformat
import numpy as np
# OS level ops
from datetime import datetime
from glob import iglob
import os
# Local modules
from pcg.speccnn import spectrogram, models
from pcg.datasets import PhysionetCinC as PCC
from pcg.metrics import ConfusionMatrix, TrainingHistory
# Misc
from pprint import pprint
# Getting the dataset in order
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import RMSprop
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, StratifiedKFold
from mlutils.classbalancing import oversample

# TODO implement command line args for this file. Num-epochs, batch size, dataset,
#       num_samples to be used for training so on and so forth

# TODO Maybe move spectrogram padding to spectrogram.py

start_time = datetime.now()
print('Starting at time', start_time)

# Get the data for training-a
print('Reading Dataset...')
data = PCC.get_subset('a')

# Compute the spectrograms and add as a column to data
print('Computing Spectrograms...')
data['spectrogram'] = [pad_sequences(s, 900, 'float32').T
                       for s in spectrogram.batch_get(data['waveform'], data['fs'][0])]

# Make train test split
# TODO Use stratification? stratify=data['label'] should work in my understanding
#       But the default is to shuffle, so statistically it shouldn't be a problem
print('Splitting dataset into test and train sets...')

hyperparameter_vectors = [
    ((900, 51)),

]

# Shuffling the dataset
data = data.sample(frac=1)


for hyperparameter_vector in hyperparameter_vectors:

    fold_id = 0
    for data_train_ids, data_test_ids in StratifiedKFold(n_splits=5).split(data, data['label']):

        start_time = datetime.now()
        print('Starting at time', start_time)
        print('For hyperparameters: ', hyperparameter_vector)
        print('Training for fold', fold_id)

        model = models.model_A(hyperparameter_vector)

        data_train = data.iloc[data_train_ids]
        data_test  = data.iloc[data_test_ids]

        # Sanity Check on shuffles
        print('Train Test Split:')
        print('Train Set:')
        print(data_train.head())
        print('Test Set')
        print(data_test.head())

        # Sanity Check on distribution
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
        model.compile(loss='binary_crossentropy',
                                optimizer=RMSprop(0.0005),
                                metrics=['accuracy'])

        history = model.fit(np.stack(data_train['spectrogram']), np.stack(data_train['label']),
                            batch_size=512,
                            epochs=50,
                            validation_data=(np.stack(data_test['spectrogram']), np.stack(data_test['label'].values)))


        # Create directory for logging
        formatted_dt = start_time.strftime('%y-%m-%d-%H-%m-%S')
        log_path = '/home/agnivesh/model_logs/speccnn/a/A/' + "{}_{} {}_{} {}".format(*hyperparameter_vector) + '/' + 'fold-{fold_id}'.format(fold_id=fold_id) + '/'
        os.makedirs(log_path)

        # Training history visualization
        TrainingHistory(history).save(log_path)

        # Saving the model configuration
        raw_json = open(log_path + 'model_architecture.json', 'w')
        raw_json.write(model.to_json())
        raw_json.close()

        pretty_json = open(log_path + 'model_description.txt', 'w')
        pretty_json.write(pformat(model.to_json()))
        pretty_json.close()

        # Save weights
        model.save_weights(log_path + 'weights.h5')

        # Save the model
        model.save(log_path + 'model.h5')

        # Confusion Matrix
        print('Calculating the Confusion Matrices...')
        train_preds = np.round(model.predict(np.stack(data_train['spectrogram']))).astype('int32')
        test_preds = np.round(model.predict(np.stack(data_test['spectrogram']))).astype('int32')

        ConfusionMatrix(data_train['label'].values, train_preds, ['Normal', 'Abnormal']).save('Train Confusion Matrix', log_path)
        ConfusionMatrix(data_test['label'].values, test_preds, ['Normal', 'Abnormal']).save('Test Confusion Matrix', log_path)

        fold_id += 1
