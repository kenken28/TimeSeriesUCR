import numpy as np
import os
import tf2onnx
import tensorflow as tf
import pandas as pd
from typing import Union


def GB(n):
    """
    Convert gigabytes into megabytes
    :param n: number of GB
    :return: number of MB
    """
    return n * 1024


def select_gpu(gpu='0', mem=1024):
    """
    Allocate a specified amount of VRam from a specified gpu device
    :param gpu: gpu device index
    :param mem: gpu memory in megabytes
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    physical_devices = tf.config.list_physical_devices('GPU')
    for device in physical_devices:
        tf.config.set_logical_device_configuration(device, [tf.config.LogicalDeviceConfiguration(memory_limit=mem)])


def record_test_accuracy(result_dir, model_name, data_name, accuracy):
    """
    Update the model accuracy table for a specified dataset
    :param result_dir: directory to store the accuracy table
    :param model_name: name of the model being used
    :param data_name: name of the dataset
    :param accuracy: model accuracy for the specified dataset
    """
    os.makedirs(result_dir, exist_ok=True)
    xlsx_path = os.path.join(result_dir, 'accuracy.xlsx')
    df_accuracy = pd.read_excel(xlsx_path) if os.path.isfile(xlsx_path) else pd.DataFrame(columns=['Dataset'])
    if model_name not in df_accuracy.columns:
        df_accuracy[model_name] = ""
    if data_name not in df_accuracy['Dataset'].unique():
        df_accuracy.loc[len(df_accuracy), ['Dataset', model_name]] = data_name, accuracy
    else:
        df_accuracy.loc[df_accuracy['Dataset'] == data_name, model_name] = accuracy
    df_accuracy.to_excel(xlsx_path, index=False)


def partitions_dataset(ds, val_portion=0.7, shuffle=True, seed=None):
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
    """
    Load a specified dataset
    :param data_dir: directory to all the dataset
    :param data_name: name of the dataset to load
    :return: numpy dataset in a dictionary
    """
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
    """
    Map the dataset label so that it starts with 0 and is continuous
    :param train_data: training dataset dictionary
    :param test_data: test dataset dictionary
    :return: an array of unique labels in the dataset
    """
    unique_train = np.unique(train_data['y'])
    unique_test = np.unique(test_data['y'])
    if not np.array_equal(unique_train, unique_test):
        raise ValueError('Training data and test data do not have equal unique label sets.')
    y_dict = {y: float(idx) for idx, y in enumerate(unique_train)}
    train_data['y'] = np.vectorize(y_dict.get)(train_data['y'])
    test_data['y'] = np.vectorize(y_dict.get)(test_data['y'])
    return unique_train


def load_tsv_to_npy(tsv_path):
    """
    Load a tsv file into a dictionary of numpy dataset
    :param tsv_path: full path to the tsv file
    :return: a dictionary of x(input) and y(label) dataset
    """
    data = np.nan_to_num(np.genfromtxt(tsv_path, dtype=float, delimiter='\t'))
    y_data = data[:, 0]
    x_data = data[:, 1:]
    return {
        'x': x_data,
        'y': y_data,
    }


def boolean_string(s):
    """
    Convert 'True' and 'False' input parameter into boolean type
    :param s: 'True' or 'False' string
    :return: True or False boolean value
    """
    if s not in ['False', 'True']:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def tf_to_onnx(model: Union[str, tf.keras.Model], save_path):
    """
    Convert a tensorFlow model into an onnx model
    :param model: a tensorFlow model object or a path string to a tensorFlow model
    :param save_path: save path for the onnx model
    :return: onnx model object
    """
    if isinstance(model, str):
        model = tf.keras.models.load_model(model)
    spec = model.input.type_spec
    spec._name = 'N_input'
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=[spec], opset=13, output_path=save_path)
    return onnx_model
