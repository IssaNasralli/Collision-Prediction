import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import model_from_json
from keras.utils import to_categorical
from keras.optimizers import Adam
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score, precision_score, recall_score
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import  Layer
from tensorflow.keras.regularizers import l2  # Import L2 regularization

# Define a custom layer to perform graph convolution
class GraphConvolution(Layer):
    def __init__(self, output_dim, l2_reg=1e-4, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.l2_reg = l2_reg  # L2 regularization parameter

    def build(self, input_shape):
        # Create a trainable weight variable for this layer
        self.kernel = self.add_weight(
            "kernel",
            (input_shape[-1], self.output_dim),
            regularizer=l2(self.l2_reg)  # Apply L2 regularization to the weights
        )

    def call(self, inputs):
        x = inputs
        output = tf.matmul(x, self.kernel)
        return output
# CNN   GRU2 GRU2_CNN LSTM   RNN   LSTM_CNN 
#   16   32   64   128    256 
#   0.01      0.001     0.0001
choice="GCN" #       #           #                      
learning_rate=0.001    #            
batch_size=64  #                 
epochs=100
plots="plots "+ str(batch_size)+" "+str(learning_rate)
# Define the custom callback class
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(ConfusionMatrixCallback, self).__init__()
        self.validation_data = validation_data
    
    def on_epoch_end(self, epoch, logs=None):
        x_val, y_val = self.validation_data
        y_pred = np.round(self.model.predict(x_val))
        cm = confusion_matrix(y_val, y_pred)
        print(f"\nConfusion Matrix after Epoch {epoch + 1}:\n{cm}")


# Read the CSV file
data_df = pd.read_csv('D4.csv')

# Extract the pose estimation values (x, y coordinates) and labels
pose_estimations = data_df.iloc[:, :-1]  # Excluding the last column (label)
labels = data_df.iloc[:, -1]  # Extract the last column (label)

# Determine the number of samples and landmarks
num_samples = len(data_df)
num_landmarks = 14

# Determine the number of frames by calculating the number of columns for each sample
num_frames = pose_estimations.shape[1] // (num_landmarks * 2)

# Convert the pose_estimations DataFrame to a 3D numpy array (X)
# Reshape the data to have num_samples samples, num_frames frames, and num_landmarks * 2 values (x, y coordinates)
X = np.array(pose_estimations).reshape(num_samples, num_frames, num_landmarks * 2)

# Convert the labels DataFrame to a numpy array (Y)
Y = np.array(labels)

print(X.shape)

# Assuming Y is the array containing class labels
class_labels, class_counts = np.unique(Y, return_counts=True)

# Print the class distribution before balancing
print("Class distribution before balancing:")
for label, count in zip(class_labels, class_counts):
    print(f"Class {label}: {count} samples")

# Reshape the 4D tensor (N, 14, 2, 16) into a 2D matrix (N, 14*2*16)
X_flat = X.reshape(num_samples, -1)

# Create a RandomUnderSampler instance
rus = RandomUnderSampler()

# Resample the data to balance the classes
X_resampled, Y_resampled = rus.fit_resample(X_flat, Y)

# Reshape the flattened X_resampled back to the original 4D tensor (N, 16, 14, 2)
X_resampled = X_resampled.reshape(-1, 16, num_landmarks * 2)
print(X_resampled.shape)

# Assuming Y_resampled is the array containing balanced class labels
class_labels_resampled, class_counts_resampled = np.unique(Y_resampled, return_counts=True)

# Print the class distribution after balancing
print("Class distribution after balancing:")
for label, count in zip(class_labels_resampled, class_counts_resampled):
    print(f"Class {label}: {count} samples")

X, Y = X_resampled, Y_resampled

# Convert the labels to one-hot encoded format
Y_one_hot = to_categorical(Y, num_classes=2)

# Load the JSON file and create the model
json_file = open('model_'+choice+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

custom_objects = {'GraphConvolution': GraphConvolution}

loaded_model = model_from_json(loaded_model_json,  custom_objects=custom_objects)

# Compile the model for binary classification
loaded_model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# Split the data into 45% training, 45% validation, and 10% testing
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.55, random_state=42, shuffle=True)
X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size=0.18, random_state=42, shuffle=True)

# Convert the data to float32 data type
X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_test = X_test.astype('float32')
output = plots+"/"+choice+"/"

# Use the loaded model for training
print("Start training...")
best_weights_path = output +choice+'_best_weights.h5'
model_checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)
tf.keras.backend.clear_session()
history = loaded_model.fit(
    X_train, Y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(X_valid, Y_valid),
    callbacks=[model_checkpoint]  # Add the ModelCheckpoint callback here
)

# Evaluate the model on the test set
loss, accuracy = loaded_model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}, Test Accuracy: {accuracy}")

# Capture training and validation metrics
training_loss = history.history['loss'][-1]
validation_loss = history.history['val_loss'][-1]
training_accuracy = history.history['accuracy'][-1]
validation_accuracy = history.history['val_accuracy'][-1]

# Get predictions for the test set
Y_pred = loaded_model.predict(X_test)

# Round the probabilities to get binary predictions
Y_pred_binary = np.round(Y_pred)

# Compute L1 score
l1_score = np.mean(np.abs(Y_test - Y_pred))

# Compute precision, recall, and average precision
precision = precision_score(Y_test, Y_pred_binary)
recall = recall_score(Y_test, Y_pred_binary)
average_precision = average_precision_score(Y_test, Y_pred)

print(f"L1 Score: {l1_score}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Average Precision: {average_precision}")

# Compute the confusion matrix
cm = confusion_matrix(Y_test, Y_pred_binary)
print(f"\nConfusion Matrix:\n{cm}")

# Plot and save the confusion matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks([0, 1], ['Standing', 'Walking'])
plt.yticks([0, 1], ['Standing', 'Walking'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
plt.savefig(output + 'confusion_matrix.png')
plt.show()

# Compute precision-recall curve
precision_curve, recall_curve, _ = precision_recall_curve(Y_test, Y_pred)

# Plot and save the precision-recall curve
plt.plot(recall_curve, precision_curve, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(output + 'precision_recall_curve.png')
plt.show()

# Save the average precision and evaluation metrics to a file
with open(output + 'average_precision.txt', 'w') as f:
    f.write("Training:\n")
    f.write(f"Validation loss at final epoch: {validation_loss:.4f}\n")
    f.write(f"Validation Accuracy at final epoch: {validation_accuracy:.4f}\n")
    f.write(f"Training loss at final epoch: {training_loss:.4f}\n")
    f.write(f"Training Accuracy at final epoch: {training_accuracy:.4f}\n")

    f.write("\nTesting:\n")
    f.write(f"Test loss: {loss:.4f}\n")
    f.write(f"Test accuracy: {accuracy:.4f}\n")
    f.write(f"L1 Score: {l1_score}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"Average Precision: {average_precision}\n")
    f.write(f"Confusion Matrix:\n{cm}")
# Access training history
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']

validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']

# Plot training and validation loss
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.savefig(output + 'Training_Validation_loss.png')
plt.show()

# Plot training and validation accuracy
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.savefig(output + 'Training_Validation_Accuracy.png')
plt.show()
