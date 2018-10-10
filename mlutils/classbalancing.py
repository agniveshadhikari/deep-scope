from sklearn.model_selection import train_test_split
from numpy import unique, random
import pandas as pd

def oversample(dataframe, column):
    classes, counts = unique(dataframe[column], return_counts=True)
    max_size = max(counts)
    indices_to_repeat = list()

    for c in classes:
        matching_df = dataframe[dataframe[column] == c]
        count = len(matching_df)
        deficit = max_size - count

        if deficit == 0:
            continue

        # Selecting 'deficit' number of random data points to repeat
        # replace=False would ensure the same sample is not added too many times,
        # but then the case where count < deficit would have to be handled separately
        indices_to_repeat_for_c = list(random.choice(matching_df.index, deficit, replace=True))
        indices_to_repeat.extend(indices_to_repeat_for_c)


    oversampled_dataframe = dataframe.append(dataframe.loc[indices_to_repeat, :], ignore_index=True)

    return oversampled_dataframe


