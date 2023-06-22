import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
from datetime import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


sys.path.append("Python3Code")
from util import util
from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning

## Set up #############################################################################################################
print("Set up")
# Define the result file
DATA_PATH = Path('Assignment/intermediate_datafiles/')
FIGURE_PATH = Path('figures/DL/')
os.makedirs(FIGURE_PATH, exist_ok=True)
DATASET_FNAME = 'final_dataset.csv'

# Load the result dataset
try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found')
    raise e


# Prepare your swimming data 
# Assuming you have two sessions, three strokes, and 5-minute duration each
# X_train and X_test should be numpy arrays of shape (num_strokes, num_timesteps, num_features)
# y_train and y_test should be numpy arrays of shape (num_strokes, num_classes)
# prepare dataset by converting labels and selecting only swimming data
prepare = PrepareDatasetForLearning()
training_set_X, test_set_X, training_set_y, test_set_y = prepare.split_single_dataset_classification(dataset, ['label'], 'like', 0.7, filter=True, temporal=True)
dataset = pd.concat([training_set_X, test_set_X])
dataset = dataset.join(pd.concat([training_set_y, test_set_y]))
dataset.index = pd.to_datetime(dataset.index)

dates = dataset.index.date
labels = dataset['class']
dataset.drop(['class'], axis=1, inplace=True)

session_dates = np.unique(dates)
stroke_types = labels.unique()

# Constants
num_sessions = len(session_dates)
num_strokes = len(stroke_types)
num_features = len(dataset.columns)
num_timepoints = util.find_max_consecutive_occurrence(labels)  # Assuming a fixed number of time points per stroke

# Reshape and pad the data
X = np.zeros((num_strokes * num_sessions, num_timepoints, num_features))
y = np.zeros(num_strokes * num_sessions, dtype=object)

# Iterate over each session
for session in range(num_sessions):
    session_start_date = session_dates[session]

    # Iterate over each stroke
    for stroke in range(num_strokes):
        # Filter data based on session and stroke
        session_stroke_data = dataset[(dates == session_start_date) & (labels == stroke_types[stroke])]
        
        # Reshape and pad the stroke data
        num_samples = len(session_stroke_data)
        padded_data = np.zeros((num_timepoints, num_features))
        
        if num_samples <= num_timepoints:
            padded_data[:num_samples] = session_stroke_data
        else:
            padded_data[:] = session_stroke_data[:num_timepoints]
        
        # Assign the reshaped and padded stroke data to the final array
        stroke_idx = session * num_strokes + stroke
        X[stroke_idx] = padded_data
        y[stroke_idx] = stroke_types[stroke]  # Assign the stroke label 

# Encode labels as integers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.flatten())

# Reshape encoded labels to a 2D array
y_reshaped = y_encoded.reshape(-1, 1)

# One-hot encode the reshaped labels
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(y_reshaped)
y_onehot = y_onehot.reshape(num_sessions * num_strokes, num_strokes)

# Split the data into training and test sets
X_train = X[:num_strokes]
X_test = X[num_strokes:]
y_train = y_onehot[:num_strokes,:]
y_test = y_onehot[num_strokes:,:]

# Create a sequential model
model = Sequential()

# Add an LSTM layer with 64 units
model.add(LSTM(64, input_shape=(num_timepoints, num_features)))

# Add a dense output layer with the desired number of classes or predictions
model.add(Dense(num_strokes, activation='softmax')) 

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model with your data
history = model.fit(X_train, y_train, epochs=10, batch_size=32)  

# Evaluate the model on your test data
loss, accuracy = model.evaluate(X_test, y_test) 

# Plot training loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(FIGURE_PATH / 'training_loss.png')
plt.show()

# Plot training accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig(FIGURE_PATH / 'training_accuracy.png')
plt.show()

# Generate predictions on test data
y_pred = model.predict(X_test)  

# Convert one-hot encoded predictions to class labels
y_pred = np.argmax(y_pred, axis=1)
# Convert one-hot encoded test labels to class labels
y_test = np.argmax(y_test, axis=1)  

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig(FIGURE_PATH / 'confusion_matrix.png')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))

