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
DATASET_FNAME = 'cleaned_dataset.csv'
RESULT_FNAME = 'final_dataset.csv'
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

print("Numerical abstraction")
for column in selected__columns:
    for ws in window_sizes:        
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'mean')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'std')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'median')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'min')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'max')
        dataset = NumAbs.abstract_numerical(dataset, [column], ws, 'energy')
    if (VERBOSE): DataViz.plot_dataset(dataset, [column, '{}_temp_mean'.format(column), '{}_temp_energy'.format(column), 'label'], ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'points'])

print("Frequency abstraction")
dataset = FreqAbs.abstract_frequency(copy.deepcopy(dataset), selected__columns, int(float(100)/milliseconds_per_instance), fs)

# Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.

print("Overlap")
# The percentage of overlap we allow
window_overlap = 0.9
skip_points = int((1-window_overlap) * ws)
dataset = dataset.iloc[::skip_points,:]

# print("Clustering")
# clusteringNH = NonHierarchicalClustering()

# dataset = clusteringNH.k_means_over_instances(dataset, ['accelerometer_X (m/s^2)', 'accelerometer_Y (m/s^2)', 'accelerometer_Z (m/s^2)'], 5, 'default', 50, 50)
# DataViz.plot_clusters_3d(dataset, ['accelerometer_X (m/s^2)', 'accelerometer_Y (m/s^2)', 'accelerometer_Z (m/s^2)'], 'cluster', ['label'])
# DataViz.plot_silhouette(dataset, 'cluster', 'silhouette')
# util.print_latex_statistics_clusters(
#     dataset, 'cluster', ['accelerometer_X (m/s^2)', 'accelerometer_Y (m/s^2)', 'accelerometer_Z (m/s^2)'], 'label')
# del dataset['silhouette']

# # Do some initial runs to determine the right number for k
# k_values = range(2, 10)
# silhouette_values = []

# for k in k_values:
#     print(f'k = {k}')
#     dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(
#         dataset), ['accelerometer_X (m/s^2)', 'accelerometer_Y (m/s^2)', 'accelerometer_Z (m/s^2)'], k, 'default', 20, 10)
#     silhouette_score = dataset_cluster['silhouette'].mean()
#     print(f'silhouette = {silhouette_score}')
#     silhouette_values.append(silhouette_score)

# DataViz.plot_xy(x=[k_values], y=[silhouette_values], xlabel='k', ylabel='silhouette score',
#                 ylim=[0, 1], line_styles=['b-'])

print("Write to file")
dataset.to_csv(DATA_PATH / RESULT_FNAME)

print("Features added",len(dataset.columns)-start_column_len)