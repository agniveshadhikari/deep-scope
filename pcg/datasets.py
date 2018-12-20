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
        'a': 409,
        'b': 480,
        'c': 31,
        'd': 55,
        'e': 2141,
        'f': 114,
    }
    num_digits = {
        'a': 4,
        'b': 4,
        'c': 4,
        'd': 4,
        'e': 5,
        'f': 4,
    }

    @staticmethod
    def _get_labels(subset):
        return pd.DataFrame(
            [
                [filename, (1 if label == '1' else 0)]
                for filename, label
                in csv.reader(open(os.path.join(PhysionetCinC.base_path,
                                                'training-' + subset,
                                                'REFERENCE.csv')))
            ][:PhysionetCinC.num_data[subset]],
            columns=['id', 'label']
        )

    @staticmethod
    def _get_data(subset):
        wav_data = list()
        for i in range(PhysionetCinC.num_data[subset]):
            data_id = subset + str(i+1).zfill(PhysionetCinC.num_digits[subset])
            wavfilepath = os.path.join(PhysionetCinC.base_path,
                                        'training-' + subset,
                                        data_id+'.wav')

            wav_data.append((data_id, *wavfile.read(wavfilepath)))

        return pd.DataFrame(wav_data, columns=['filename', 'fs', 'waveform'])

    @staticmethod
    def get_subset(subset):

        # Load the labels
        # cols: id, label
        labels = PhysionetCinC._get_labels(subset)

        # Read the data files
        # cols: filename, fs, waveform
        wav_data = PhysionetCinC._get_data(subset)

        # Construct a dataframe and return.
        # Filename and ID from labels are reduntdant, but added as sanity check
        return pd.DataFrame({
            'id': labels.id,
            'filename': wav_data.filename,
            'waveform': wav_data.waveform,
            'label': labels.label,
            'fs': wav_data.fs
        })

    @staticmethod
    def get_all_subsets():
        combined = pd.concat([
                PhysionetCinC.get_subset(subset) for subset in PhysionetCinC.num_data
            ],
            ignore_index=True
        )
        return combined

if __name__ == '__main__':
    print('The following is returned for the call: PhysionetCinC.get_subset(\'a\')')
    print(PhysionetCinC.get_subset('a'))