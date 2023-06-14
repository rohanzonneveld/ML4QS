import sys, copy
import pandas as pd
import time
from pathlib import Path
import argparse

sys.path.append("Python3Code")

from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.TemporalAbstraction import CategoricalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter4.TextAbstraction import TextAbstraction

DATA_PATH = Path('Assignment/intermediate_datafiles/')
DATASET_FNAME = 'dataset.csv'
RESULT_FNAME = 'dataset_features.csv'

dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
dataset.index = pd.to_datetime(dataset.index)

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = 1000

window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(5*60000)/milliseconds_per_instance)]
ws = int(float(0.5*60000)/milliseconds_per_instance)
fs = float(1000)/milliseconds_per_instance

DataViz = VisualizeDataset(__file__)

NumAbs = NumericalAbstraction()
FreqAbs = FourierTransformation()
        
        
selected_predictor_cols = [c for c in dataset.columns if not 'label' in c]

#columns = ['accelerometer_X (m/s^2)', 'accelerometer_Y (m/s^2)', 'accelerometer_Z (m/s^2)']

for column in selected_predictor_cols:
    for ws in window_sizes:        
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'std')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'median')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'min')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'max')


#dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), selected_predictor_cols, int(float(10000)/milliseconds_per_instance), fs)


# Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

# The percentage of overlap we allow
window_overlap = 0.9
skip_points = int((1-window_overlap) * ws)
dataset = dataset.iloc[::skip_points,:]


dataset.to_csv(DATA_PATH / RESULT_FNAME)

DataViz.plot_dataset(dataset, selected_predictor_cols, ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'points'])

