import tensorflow as tf


def create_mlp(in_size, out_size, hidden_layer_nodes):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=in_size),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(hidden_layer_nodes, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_nodes // 2, activation='relu'),
        tf.keras.layers.Dense(hidden_layer_nodes // 4, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(out_size),
    ])
    return model


def create_conv(input_size, out_size, kernel, stride, drop_rate=0.5):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_size),
        tf.keras.layers.Reshape((input_size, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=8, kernel_size=kernel, strides=stride, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=32, kernel_size=kernel, strides=stride, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(filters=128, kernel_size=kernel, strides=stride, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(drop_rate),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(out_size),
    ])
    return model


def create_lstm(input_size, out_size, drop_rate=0.5):
    x_in = tf.keras.layers.Input(shape=input_size)
    x_0 = tf.keras.layers.Reshape((1, input_size))(x_in)
    x = tf.keras.layers.LSTM(128)(x_0)
    x = tf.keras.layers.Dropout(drop_rate)(x)

    y = tf.keras.layers.Permute((2, 1))(x_0)
    y = tf.keras.layers.Conv1D(128, 8, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)

    y = tf.keras.layers.Conv1D(256, 5, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)

    y = tf.keras.layers.Conv1D(128, 3, padding='same')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y = tf.keras.layers.GlobalAveragePooling1D()(y)
    y = tf.keras.layers.Dropout(drop_rate)(y)

    z = tf.keras.layers.concatenate([x, y])
    out = tf.keras.layers.Dense(out_size)(z)

    model = tf.keras.Model(x_in, out)
    return model
