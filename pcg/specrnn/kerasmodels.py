from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, Dropout, BatchNormalization

model = Sequential(
    layers=[
        BatchNormalization(),
        LSTM(100, input_shape=(900, 51), return_sequences=True, dropout=0.0),
        # Dropout(0.5),
        LSTM(20, dropout=0.0),
        # Dropout(0.4),
        Dense(1, activation='sigmoid')
    ]
)


# TODO Implement functions that take args the hyperparameters and returns the model.
#       This way, models with same architectures but different hyparams, and different
#       architectures won't get mixed up. see models.py

models = [
    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(100, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(10, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),

    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(200, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(10, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),

    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(300, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(10, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),

    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(100, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(20, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),

    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(200, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(20, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),

    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(300, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(20, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),

    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(100, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(30, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),

    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(200, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(30, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),

    Sequential(
        layers=[
            BatchNormalization(input_shape=(900, 51)),
            LSTM(300, input_shape=(900, 51), return_sequences=True, dropout=0.0),
            LSTM(30, dropout=0.0),
            Dense(1, activation='sigmoid')
        ]
    ),
]