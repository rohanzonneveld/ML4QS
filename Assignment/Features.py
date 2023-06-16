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
DATASET_FNAME = 'small_cleaned_dataset.csv'
RESULT_FNAME = 'small_final_dataset.csv'
VERBOSE = False
dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
dataset.sort_index(inplace=True)
#dataset = dataset.head(30_000)

dataset.index = pd.to_datetime(dataset.index)

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = 10.0

window_sizes = [int(float(5000)/milliseconds_per_instance), int(float(0.5*60000)/milliseconds_per_instance), int(float(5*60000)/milliseconds_per_instance)]
ws = int(float(0.5*60000)/milliseconds_per_instance)
fs = float(1000)/milliseconds_per_instance

DataViz = VisualizeDataset(__file__)

NumAbs = NumericalAbstraction()
FreqAbs = FourierTransformation()

start_column_len = len(dataset.columns)

selected__columns = [c for c in dataset.columns if not 'label' in c]

for column in selected__columns:
    for ws in window_sizes:        
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'std')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'median')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'min')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'max')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'energy')
    if (VERBOSE): DataViz.plot_dataset(dataset, [column, '{}_temp_mean'.format(column), '{}_temp_energy'.format(column), 'label'], ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])

dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), selected__columns, int(float(100)/milliseconds_per_instance), fs)

# Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

# The percentage of overlap we allow
window_overlap = 0.9
skip_points = int((1-window_overlap) * ws)
dataset = dataset.iloc[::skip_points,:]

dataset.to_csv(DATA_PATH / RESULT_FNAME)

print("Features added",len(dataset.columns)-start_column_len)