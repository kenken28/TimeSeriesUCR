import numpy as np
import os
import tf2onnx
import tensorflow as tf
from typing import Union


def GB(n):
    return n * 1024


def select_gpu(gpu='7', mem=1024):
    import os
    import tensorflow as tf
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])


def partitions_dataset(ds, val_portion=0.7, shuffle=True, seed=0):
    """
    Partition the given dataset into training, validation, and test sets
    :param ds: inout & output dataset in tf.data.Dataset format
    :param val_portion: percentage of validation dataset
    :param shuffle: set to True to shuffle dataset before splitting
    :param seed: seed for shuffling
    :return: a tuple of (training, validation) datasets
    """
    assert val_portion < 1.0
    ds_size = ds.cardinality().numpy()
    if shuffle:
        ds = ds.shuffle(ds_size, seed=seed)
    # split datasets
    val_size = round(val_portion * ds_size)
    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)
    return train_ds, val_ds


def load_data(data_dir, data_name):
    train_data_path = os.path.join(data_dir, data_name, f'{data_name}_TEST.tsv')
    test_data_path = os.path.join(data_dir, data_name, f'{data_name}_TRAIN.tsv')
    train_data = load_tsv_to_npy(train_data_path)
    test_data = load_tsv_to_npy(test_data_path)
    unique_y_list = map_y(train_data, test_data)
    return {
        'train': train_data,
        'test': test_data,
    }, unique_y_list


def map_y(train_data, test_data):
    unique_train = np.unique(train_data['y'])
    unique_test = np.unique(test_data['y'])
    if not np.array_equal(unique_train, unique_test):
        raise ValueError('Training data and test data do not have equal unique label sets.')
    y_dict = {y: float(idx) for idx, y in enumerate(unique_train)}
    train_data['y'] = np.vectorize(y_dict.get)(train_data['y'])
    test_data['y'] = np.vectorize(y_dict.get)(test_data['y'])
    return unique_train


def load_tsv_to_npy(tsv_path):
    data = np.nan_to_num(np.genfromtxt(tsv_path, dtype=float, delimiter='\t'))
    y_data = data[:, 0]
    x_data = data[:, 1:]
    return {
        'x': x_data,
        'y': y_data,
    }


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def tf_to_onnx(model: Union[str, tf.keras.Model], save_path):
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)
    spec = model.input.type_spec
    spec._name = 'N_input'
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[spec], opset=13, output_path=save_path)
    return onnx_model
