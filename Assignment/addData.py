import sys, os
import pandas as pd

sys.path.append("Python3Code")
from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util

from pathlib import Path
import os

def get_folders_in_path(path):
    folders = []
    for entry in os.scandir(path):
        if entry.is_dir():
            folders.append(entry.name)
    return folders

DATASET_PATH = Path('Assignment/data/phone/')
RESULT_PATH = Path('Assignment/intermediate_datafiles/')
RESULT_FNAME = 'dataset.csv'

[path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

granularity = 10
dataset = pd.DataFrame()

folders = get_folders_in_path(DATASET_PATH)
for folder in folders:
    dummy = CreateDataset(DATASET_PATH, granularity)
    start_time = pd.read_csv(DATASET_PATH / folder / 'meta/time.csv')['system time'][0]
    dummy.add_numerical_dataset(folder + '/Barometer.csv', 'Time (s)', ['X (hPa)'], 'avg', 'barometer_', start_time=start_time)
    dummy.add_numerical_dataset(folder + '/Gyroscope.csv', 'Time (s)', ['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'], 'avg', 'gyroscope_', start_time=start_time)
    dummy.add_numerical_dataset(folder + '/Linear Accelerometer.csv', 'Time (s)', ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'], 'avg', 'accelerometer_', start_time=start_time)
    dummy.add_numerical_dataset(folder + '/Location.csv', 'Time (s)', ['Velocity (m/s)'], 'avg', 'location_', start_time=start_time)
    dummy.add_event_dataset(folder + '/labels.csv', 'label_start', 'label_end', 'label', 'binary', start_time=start_time)
    # stack dummy on top of dataset
    dataset = pd.concat([dataset, dummy.data_table])


util.print_statistics(dataset)
# Plot the data
DataViz = VisualizeDataset(__file__)
#DataViz.plot_dataset_boxplot(dataset, ['gyroscope_X (rad/s)','gyroscope_Y (rad/s) ','gyroscope_Z (rad/s) ','accelerometer_X (m/s^2)','accelerometer_Y (m/s^2)','accelerometer_Z (m/s^2)'])
# DataViz.plot_dataset_boxplot(dataset, ['accelerometer_X (m/s^2)', 'accelerometer_Y (m/s^2)', 'accelerometer_Z (m/s^2)', 'gyroscope_X (rad/s)', 'gyroscope_Y (rad/s)', 'gyroscope_Z (rad/s)'])

# Plot all data
dataset.to_csv(RESULT_PATH / RESULT_FNAME)