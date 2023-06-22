import sys, copy
import pandas as pd
import time
from pathlib import Path
import numpy as np
sys.path.append("Python3Code")
import util.util as util
from util.VisualizeDataset import VisualizeDataset
from Chapter4.TemporalAbstraction import NumericalAbstraction
from Chapter4.FrequencyAbstraction import FourierTransformation
from Chapter5.Clustering import NonHierarchicalClustering

print("Set up")
DATA_PATH = Path('Assignment/intermediate_datafiles/')
DATASET_FNAME = 'final_dataset.csv'
RESULT_FNAME = 'test_final_dataset.csv'
VERBOSE = False
dataset = pd.read_csv(DATA_PATH / DATASET_FNAME, index_col=0)
dataset.sort_index(inplace=True)

dataset.index = pd.to_datetime(dataset.index)

# Compute the number of milliseconds covered by an instance based on the first two rows
milliseconds_per_instance = 10
ws = int(float(100)/milliseconds_per_instance)

# Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.
print(dataset.shape)
print("Overlap")
# The percentage of overlap we allow
window_overlap = 0.9
skip_points = int((1-window_overlap) * ws)
dataset = dataset.iloc[::skip_points,:]

print(dataset.shape)

print("Write to file")
dataset.to_csv(DATA_PATH / RESULT_FNAME)

