from keras.models import Sequential
from keras.layers import Activation, LSTM, Dense, Dropout

model = Sequential(
    layers=[
        LSTM(100, input_shape=(165, 251)),
        # Dropout(0.2),
        Dense(1, activation='sigmoid')
    ]
)