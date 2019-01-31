import os
from datetime import datetime
from pprint import pformat

import numpy as np
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle

from mlutils.classbalancing import oversample
from pcg.datasets import PhysionetCinC as PCC
from pcg.metrics import ConfusionMatrix, TrainingHistory
from pcg.mfccrnn import models, mfcc


SCRIPT_START_TIME = datetime.now()
print('Starting at time', SCRIPT_START_TIME)

# Get the data for training-a
print('Reading Dataset...')
data = PCC.get_subset('a')

# Compute the spectrograms and add as a column to data
print('Computing Spectrograms...')
data['spectrogram'] = [pad_sequences(s, 800, 'float32').T
                       for s in mfcc.MFCC(y=data['waveform'],
                                          sampling_rate=data['fs'][0],
                                          nperseg=100,
                                          stride=50,
                                          n_mfcc=20)
                                .get()]

hyperparameter_vectors = [

    (30, 10, 0.5, 0.4, (800, 20)),
    (30, 10, 0.6, 0.4, (800, 20)),
    (30, 10, 0.7, 0.4, (800, 20)),
    (30, 10, 0.8, 0.4, (800, 20)),
    (30, 10, 0.9, 0.4, (800, 20)),
    (30, 10, 0.95, 0.4, (800, 20)),

    (30, 10, 0.5, 0.6, (800, 20)),
    (30, 10, 0.6, 0.6, (800, 20)),
    (30, 10, 0.7, 0.6, (800, 20)),
    (30, 10, 0.8, 0.6, (800, 20)),
    (30, 10, 0.9, 0.6, (800, 20)),
    (30, 10, 0.95, 0.6, (800, 20)),

    (30, 10, 0.5, 0.8, (800, 20)),
    (30, 10, 0.6, 0.8, (800, 20)),
    (30, 10, 0.7, 0.8, (800, 20)),
    (30, 10, 0.8, 0.8, (800, 20)),
    (30, 10, 0.9, 0.8, (800, 20)),
    (30, 10, 0.95, 0.8, (800, 20)),


    (15, 10, 0.5, 0.5, (800, 20)),
    (15, 10, 0.6, 0.5, (800, 20)),
    (15, 10, 0.7, 0.5, (800, 20)),
    (15, 10, 0.8, 0.5, (800, 20)),
    (15, 10, 0.9, 0.5, (800, 20)),

    (10, 10, 0.5, 0.5, (800, 20)),
    (10, 10, 0.6, 0.5, (800, 20)),
    (10, 10, 0.7, 0.5, (800, 20)),
    (10, 10, 0.8, 0.5, (800, 20)),
    (10, 10, 0.9, 0.5, (800, 20)),

    (15, 5, 0.5, 0.4, (800, 20)),
    (15, 5, 0.6, 0.4, (800, 20)),
    (15, 5, 0.7, 0.4, (800, 20)),
    (15, 5, 0.8, 0.4, (800, 20)),
    (15, 5, 0.9, 0.4, (800, 20)),

    (10, 5, 0.5, 0.4, (800, 20)),
    (10, 5, 0.6, 0.4, (800, 20)),
    (10, 5, 0.7, 0.4, (800, 20)),
    (10, 5, 0.8, 0.4, (800, 20)),
    (10, 5, 0.9, 0.4, (800, 20)),

    (5, 3, 0, 0, (800, 20)),
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

        model = models.model_C(*hyperparameter_vector)

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
                            epochs=300,
                            validation_data=(np.stack(data_test['spectrogram']), np.stack(data_test['label'].values)))


        # Create directory for logging
        formatted_dt = start_time.strftime('%y-%m-%d-%H-%m-%S')
        log_path = '/home/agnivesh/model_logs/mfccrnn/a/C/' + "{}_{} {}_{} {}".format(*hyperparameter_vector) + '/' + 'fold-{fold_id}'.format(fold_id=fold_id) + '/'
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
