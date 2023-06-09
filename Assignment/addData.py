import sys
sys.path.append('../Python3Code')
from Python3Code.Chapter2.CreateDataset import CreateDataset
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

dataset = CreateDataset(DATASET_PATH, 10)

folders = get_folders_in_path(DATASET_PATH)
for folder in folders:
    dataset.add_numerical_dataset(folder + '/Barometer.csv', 'Time (s)', ['X (hPa)'], 'avg', 'barometer_')
    dataset.add_numerical_dataset(folder + '/Gyroscope.csv', 'Time (s)', ['X (rad/s)', 'Y (rad/s)', 'Z (rad/s)'], 'avg', 'gyroscope_')
    dataset.add_numerical_dataset(folder + '/Accelerometer.csv', 'Time (s)', ['X (m/s^2)', 'Y (m/s^2)', 'Z (m/s^2)'], 'avg', 'accelerometer_')
    dataset.add_numerical_dataset(folder + '/Location.csv', 'Time (s)', ['Velocity (m/s)'], 'avg', 'location_')
    dataset.add_event_dataset(folder + '/labels.csv', 'label_start', 'label_end', 'label', 'binary')

dataset.to_csv(RESULT_PATH / RESULT_FNAME)