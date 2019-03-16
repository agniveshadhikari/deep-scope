from keras import Model, Sequential
from keras.layers import Conv1D, MaxPool1D, LSTM, Dense, BatchNormalization

def model_A(input_shape):
    return Sequential(layers=[
        BatchNormalization(input_shape=input_shape),

        Conv1D(8, kernel_size=5, strides=1),
        Conv1D(16, kernel_size=5, strides=1),
        Conv1D(16, kernel_size=5, strides=1),
        MaxPool1D(pool_size=2),

        Conv1D(32, kernel_size=5, strides=1),
        Conv1D(32, kernel_size=5, strides=1),
        MaxPool1D(pool_size=2),

        Conv1D(64, kernel_size=5, strides=1),
        Conv1D(64, kernel_size=5, strides=1),
        MaxPool1D(pool_size=2),

        LSTM(20, return_sequences=True),
        LSTM(10),
        Dense(1, activation='sigmoid')
    ])