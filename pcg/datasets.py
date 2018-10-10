import pandas as pd
import numpy as np
from scipy.io import wavfile
import os
import csv

# TODO Get this whole file in shape. This is going to be a major PITA

# TODO Decide on a scheme of class equalization.
#       - When using a single set
#       - When using all sets

# In my opinion, use manual per subset settings and one overall setting


class PhysionetCinC:
    base_path = "/home/agnivesh/Documents/fyp/physionet-cinc/Dataset/training"
    num_data = {
        'training-a': 400  # TODO Currently ignoring data post 400
    }

    @staticmethod
    def _get_labels(subset='training-a'):
        return pd.DataFrame(
            [
                (filename, 1 if label == '1' else 0)
                for filename, label
                in csv.reader(open(os.path.join(PhysionetCinC.base_path,
                                                subset,
                                                'REFERENCE.csv')))
            ][:PhysionetCinC.num_data[subset]],
            columns=['id', 'label']
        )

    @staticmethod
    def _get_data(subset='training-a'):
        wav_data = list()
        for i in range(PhysionetCinC.num_data[subset]):
            data_id = 'a'+str(i+1).zfill(4)
            wavfilepath = os.path.join(PhysionetCinC.base_path,
                                        'training-a',
                                        data_id+'.wav')

            wav_data.append(wavfile.read(wavfilepath))

        return pd.DataFrame(wav_data, columns=['fs', 'waveform'])

    @staticmethod
    def get_subset(subset='training-a'):

        # Initialize an empty pandas.DataFrame which is going to be populated
        # and finally returned

        # Load the labels
        # cols: id, label
        labels = PhysionetCinC._get_labels(subset)

        # Read the data files
        # cols: fs, waveform
        wav_data = PhysionetCinC._get_data(subset)

        # Stack the wav data and the labels horizontally
        # cols: id, label, fs, waveform
        # raw_data = np.hstack([wav_data, labels])

        # Append this subset's data to the dataframe to be returned
        return pd.DataFrame([labels.id, wav_data.waveform, labels.label, wav_data.fs]).T

    @staticmethod
    def get_all_subsets():
        pass

if __name__ == '__main__':
    print('The following is returned for the call: PhysionetCinC.get_subset(\'training-a\')')
    print(PhysionetCinC.get_subset())