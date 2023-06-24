import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tcn import TCN, tcn_full_summary
from pathlib import Path

num_sessions = 2
num_strokes = 3
num_features = 200
num_timepoints = 10*100  

learning_rate = 3.306727369753286e-05
num_filters = 64
kernel_size = 10
dropout_rate = 0.0011129628019615496
dilations = [1, 2, 4, 8, 16, 32]

inputs = Input(shape=(num_timepoints, num_features))
x = TCN(num_filters, kernel_size, dilations=dilations, dropout_rate=dropout_rate, return_sequences=False)(inputs)
outputs = Dense(num_strokes, activation='relu')(x)
model = Model(inputs=[inputs], outputs=[outputs])

FIGURE_PATH = Path('figures/DL_TCN_1/')
tf.keras.utils.plot_model(
    model,
    to_file=FIGURE_PATH/'TCN_model.png',
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    rankdir='TB',
    dpi=200,
    layer_range=None,
)