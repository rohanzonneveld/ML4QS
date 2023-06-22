import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, MaxPooling1D, Conv1D, Dropout
from tensorflow.keras.optimizers.legacy import Adam
import optuna
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


sys.path.append("Python3Code")
from util import util
from util.VisualizeDataset import VisualizeDataset
from Chapter7.PrepareDatasetForLearning import PrepareDatasetForLearning

## Set up #############################################################################################################
# Define the result file
DATA_PATH = Path('Assignment/intermediate_datafiles/')
DATASET_FNAME = 'final_dataset.csv'

# Load the result dataset
print('Loading data...')
try:
    dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
except IOError as e:
    print('File not found')
    raise e


# Prepare your swimming data 
print('Preparing data...')
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

def objective(trial):
    # Define the parameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    num_filters = trial.suggest_categorical('num_filters', [32, 64, 128, 256])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    epochs = trial.suggest_categorical('epochs', [10, 20, 30, 40, 50])
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    # Build the model
    model = Sequential()
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', input_shape=(num_timepoints, num_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_strokes, activation='softmax'))

    # Compile the model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)

    # Return the validation accuracy as the objective value
    return accuracy

# Create the Optuna study
study = optuna.create_study(direction='maximize')

# Optimize the objective function
study.optimize(objective, n_trials=100)

# Print the best trial's parameters and objective value
best_trial = study.best_trial
print(f'Best Accuracy: {best_trial.value:.4f}')
print(f'Best Parameters: {best_trial.params}')