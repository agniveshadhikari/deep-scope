from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, Dropout, BatchNormalization, CuDNNLSTM
from keras.layers.wrappers import Bidirectional


def model_A(lstm_1, lstm_2, dropout_1, dropout_2, input_shape):
    """
    Model A Architecture:
    BatchNorm
    LSTM_1
    LSTM_2
    Dense
    """

    return Sequential(
        layers=[
            BatchNormalization(input_shape=input_shape),
            LSTM(lstm_1, return_sequences=True, dropout=dropout_1),
            LSTM(lstm_2, dropout=dropout_2),
            Dense(1, activation='sigmoid')
        ]
    )


def model_B(lstm_1, lstm_2, dropout_1, dropout_2, input_shape):
    """
    Model B Architecture:
    BatchNorm
    LSTM_1
    LSTM_2
    Dense
    """

    return Sequential(
        layers=[
            BatchNormalization(input_shape=input_shape),
            LSTM(lstm_1, return_sequences=True, dropout=dropout_1),
            BatchNormalization(),
            LSTM(lstm_2, dropout=dropout_2),
            Dense(1, activation='sigmoid')
        ]
    )


def model_C(lstm_1, lstm_2, dropout_1, dropout_2, input_shape):
    """
    Model B Architecture:
    BatchNorm
    LSTM_1
    LSTM_2
    Dense
    """

    return Sequential(
        layers=[
            BatchNormalization(input_shape=input_shape),
            Bidirectional(CuDNNLSTM(lstm_1, return_sequences=True)),
            Dropout(dropout_1),
            BatchNormalization(),
            Bidirectional(CuDNNLSTM(lstm_2)),
            Dropout(dropout_2),
            Dense(1, activation='sigmoid')
        ]
    )
