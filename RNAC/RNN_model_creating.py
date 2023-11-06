from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Layer, Reshape, MaxPooling1D, Bidirectional, Dropout,Dense, SimpleRNN , Conv2D, MaxPooling2D, Flatten, LSTM,GRU,BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.layers import Concatenate
from keras.layers import Lambda
import tensorflow as tf



class GraphConvolution(Layer):
    def __init__(self, units, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1].as_list()[0], self.units),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs):
        features, adjacency_matrix = inputs
        output = tf.matmul(adjacency_matrix, features)
        output = tf.matmul(output, self.kernel)
        return tf.nn.relu(output)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], self.units

def create_gcn_model(input_shape, adj_matrix):
    num_filters = 64

    # Input Layer
    input_tensor = Input(shape=input_shape)
    output_tensor=""

    # Reshape the input tensor to be (batch_size * num_frames, num_keypoints * num_coordinates)
    reshaped_input = tf.reshape(input_tensor, (-1, input_shape[1] * input_shape[2]))

    # Convert adjacency matrix to TensorFlow tensor
    adj_matrix_tensor = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

    # GCN Layer with the correct number of units (filters)
    gcn_output = GraphConvolution(num_filters)([reshaped_input, adj_matrix_tensor])

    # Reshape back to (batch_size, num_frames, num_keypoints, num_filters)
    gcn_output = tf.reshape(gcn_output, (-1, input_shape[1], input_shape[2], num_filters))

    # Global Average Pooling (GAP)
    pooled_output = tf.reduce_mean(gcn_output, axis=1)

    # Fully Connected Layer
    fc_output = Dense(64, activation='relu')(pooled_output)

    # Output layer with sigmoid activation for binary classification
    output_tensor = Dense(1, activation='sigmoid')(fc_output)

    model = Model(inputs=input_tensor, outputs=output_tensor)
    return 

def create_model(choice,input_shape,adj_matrix):
    if choice == "gcn":
        model = create_gcn_model(input_shape, adj_matrix)
    elif choice == "rnn":
        model = Sequential()
        model.add(SimpleRNN(12, input_shape=(16, num_keypoints * 2)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    elif choice == "lstm":
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(LSTM(64))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    elif choice == "gru1":
        model = Sequential()
        model.add(GRU(12, input_shape=(16, num_keypoints * 2)))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    elif choice == "gru2":
        model = Sequential()
        model.add(GRU(64, input_shape=(16, num_keypoints * 2), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # Add the rest of the layers for GRU2...
    elif choice == "gru_cnn":
        model = Sequential()
        model.add(GRU(64, input_shape=(16, num_keypoints * 2), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # Add the rest of the layers for GRU_CNN...
    elif choice == "lstm_cnn":
        model = Sequential()
        model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # Add the rest of the layers for LSTM_CNN...
        model.add(Conv1D(32, kernel_size=3, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        # Pooling layer to reduce the sequence length
        model.add(MaxPooling1D(pool_size=2))
        # Flatten the output before feeding it to dense layers
        model.add(Flatten())
        # Dense layers with dropout and batch normalization
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        # Output layer with sigmoid activation for binary classification
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    else:
        raise ValueError("Invalid choice. Supported choices are: 'gcn', 'rnn', 'lstm', 'gru1', 'gru2', 'gru_cnn', 'lstm_cnn'")
    return model                                               
np.random.seed(7)

num_keypoints=14
model = Sequential()
input_shape = (16, num_keypoints*2)
input_shape_gcn = (16, num_keypoints, 2)
# Example adjacency matrix (16x16 for 16 frames)
num_frames = 16
adj_matrix = np.zeros((num_frames, num_frames))
for i in range(num_frames - 1):
    adj_matrix[i, i + 1] = 1
    adj_matrix[i + 1, i] = 1


choice = "lstm_cnn"
model = create_model(choice, input_shape, adj_matrix)
model.summary()

model_json = model.to_json()
with open("model_lstm_cnn.json", "w") as json_file:
    json_file.write(model_json)

