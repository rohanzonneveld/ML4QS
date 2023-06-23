import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from tcn import TCN, tcn_full_summary
import optuna
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split


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
num_timepoints = 10*100  # 10 seconds of data sampled at 100 Hz

# Reshape and pad the data
max_batch = int(util.find_max_consecutive_occurrence(labels)/num_timepoints) # the maximum number of batches of num_timepoints that can be extracted from the data
X = np.zeros((num_strokes * num_sessions * max_batch, num_timepoints, num_features))
y = np.zeros(num_strokes * num_sessions * max_batch, dtype=object)

# Iterate over each session
idx = 0
for session in range(num_sessions):
    session_start_date = session_dates[session]

    # Iterate over each stroke
    for stroke in range(num_strokes):
        # Filter data based on session and stroke
        session_stroke_data = dataset[(dates == session_start_date) & (labels == stroke_types[stroke])]
        
        # Reshape and pad the stroke data
        num_samples = len(session_stroke_data)
        batches = int(np.floor(num_samples / num_timepoints))
        for batch in range(batches):
            start_idx = batch * num_timepoints
            end_idx = (batch + 1) * num_timepoints
            if end_idx > num_samples:
                break
            X[idx, :, :] = session_stroke_data.iloc[start_idx:end_idx].values
            y[idx] = stroke_types[stroke]  
            idx += 1

# delete all rows from X and y that are all zeros
X = X[~np.all(X == 0, axis=(1, 2))]
y = y[~(y.astype(str) == '0')]

# Encode labels as integers using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.flatten())

# Reshape encoded labels to a 2D array
y_reshaped = y_encoded.reshape(-1, 1)

# One-hot encode the reshaped labels
onehot_encoder = OneHotEncoder(sparse=False)
y_onehot = onehot_encoder.fit_transform(y_reshaped)
y_onehot = y_onehot.reshape(-1, num_strokes)

# Split the data into training and test sets in a stratified manner
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3, stratify=y_onehot)

def objective(trial):
    # Define the parameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-1, log=True)
    num_filters = trial.suggest_categorical('num_filters', [32, 64, 128, 256])
    kernel_size = trial.suggest_categorical('kernel_size', [3, 5, 10])
    dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
    dilations = trial.suggest_categorical('dilations', [[1, 2, 4, 8, 16, 32, 64], [1, 2, 4, 8, 16, 32], [1, 2, 4, 8, 16], [1, 2, 4, 8], [1, 2, 4], [1, 2], [1]])

    # Build the model
    inputs = Input(shape=(num_timepoints, num_features))
    x = TCN(num_filters, kernel_size, dilations=dilations, dropout_rate=dropout_rate, return_sequences=False)(inputs)
    outputs = Dense(num_strokes, activation='relu')(x)
    model = Model(inputs=[inputs], outputs=[outputs])

    # Compile the model
    model.compile(Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

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