from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, Dropout, BatchNormalization

model = Sequential(
    layers=[
        BatchNormalization(),
        LSTM(30, input_shape=(900, 51), return_sequences=True),
        Dropout(0.5),
        LSTM(10),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ]
)