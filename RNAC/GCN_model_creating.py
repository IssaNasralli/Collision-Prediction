import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Activation, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.regularizers import l2


# Define batch size and input shape
time_steps = 16
num_landmarks = 28
input_shape = (time_steps, num_landmarks)

# Create the adjacency matrix based on the batch size
given_matrix = [[1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,1,1,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,1,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1,0,0,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,1,0,0,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,1,0,0,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,1,0,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,1,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1]
    ]

adjacency_matrix = np.array(given_matrix, dtype=np.float32)

# Define a custom layer to perform graph convolution
class GraphConvolution(Layer):
    def __init__(self, output_dim, regularization_strength=0.0001, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.regularization_strength = regularization_strength

    def build(self, input_shape):
        # Create a trainable weight variable for this layer
        self.kernel = self.add_weight(
            "kernel",
            (input_shape[-1], self.output_dim),
            initializer='glorot_uniform',  # You can choose an initializer
            regularizer=l2(self.regularization_strength)  # Set the regularization strength here
        )

    def call(self, inputs):
        x = inputs
        output = tf.matmul(x, self.kernel)
        return output

# Define the input tensor
input_tensor = Input(shape=input_shape)

# Reshape input tensor to be compatible with GraphConvolution
x = Reshape((time_steps * num_landmarks,))(input_tensor)

# Apply custom GraphConvolution layers
x = GraphConvolution(64)(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.5)(x)
x = GraphConvolution(32)(x)
x = LeakyReLU(alpha=0.1)(x)
x = Dropout(0.5)(x)

# Output layer
output_tensor = Dense(1, activation='sigmoid')(x)

# Create the model
gcn_model = Model(inputs=input_tensor, outputs=output_tensor)

# Compile the model
gcn_model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
gcn_model.summary()

model_json = gcn_model.to_json()
with open("model_gcn.json", "w") as json_file:
    json_file.write(model_json)