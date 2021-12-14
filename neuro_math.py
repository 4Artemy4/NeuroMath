import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class NeuroMath:
    seq = []
    window_size = 0
    model = None
    epoch_count = 500

    def __init__(self, input_seq, window_size=3, epoch_count=500):
        self.features_count = 1
        self.seq = input_seq
        self.window_size = window_size
        self.epoch_count = epoch_count

    def split_sequence(self):
        x = []
        y = []

        for i in range(len(self.seq)):
            last_index = i + self.window_size
            if last_index > len(self.seq) - 1:
                break
            seq_x, seq_y = self.seq[i:last_index], self.seq[last_index]
            x.append(seq_x)
            y.append(seq_y)
            pass
        x = np.array(x)
        y = np.array(y)
        return x, y

        pass

    def training(self):
        # print input sequence
        print(self.seq)
        x, y = self.split_sequence()
        # print split sequence
        print(x)
        x = x.reshape((x.shape[0], x.shape[1], self.features_count))
        self.model = tf.keras.Sequential()
        self.model.add(layers.LSTM(50, activation='relu', input_shape=(self.window_size, self.features_count)))
        self.model.add(layers.Dense(1))
        self.model.summary()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.01), loss=tf.keras.losses.MeanSquaredError())
        self.model.fit(x, y, epochs=self.epoch_count, verbose=0)

    def guessing(self, test_data):
        test_data = test_data.reshape((1, self.window_size, self.features_count))
        predict_next_number = self.model.predict(test_data, verbose=0)
        return predict_next_number
