import numpy as np
import os
import argparse
import tensorflow as tf
from utils import *
select_gpu(0, GB(4))


class CheckpointSaver(tf.keras.callbacks.Callback):
    """
    Callback for saving model checkpoints with the best performance (in terms of loss and accuracy)
    """
    def __init__(self, cp_dir, prefix, suffix, save_per_n_rounds=1):
        self.best_acc = 0.0
        self.best_loss = 9999999.9
        self.best_acc_fn = None
        self.best_loss_fn = None
        self.prefix = prefix
        self.suffix = suffix
        self.cp_dir = cp_dir
        self.save_per_n_rounds = save_per_n_rounds

    def reset(self):
        self.best_acc = 0.0
        self.best_loss = 9999999.9
        self.best_acc_fn = None
        self.best_loss_fn = None

    def gen_filepath(self, attr, epoch):
        best_val = self.best_acc if attr == 'acc' else self.best_loss
        fn = '{}_{}{:.4f}_e{}_{}'.format(self.prefix, attr, best_val, epoch, self.suffix)
        return os.path.join(self.cp_dir, fn)

    def save_onnx(self, onnx_save_path):
        tf_to_onnx(self.model, onnx_save_path)

    def save_model(self, attr, epoch):
        fn = self.gen_filepath(attr, epoch)
        onnx_fn = os.path.join(fn, f'{os.path.basename(fn)}.onnx')
        self.model.save(fn, save_format='tf')
        self.save_onnx(onnx_fn)
        # delete old checkpoint
        if attr == 'acc':
            self.delete_model(self.best_acc_fn)
            self.best_acc_fn = fn
        else:
            self.delete_model(self.best_loss_fn)
            self.best_loss_fn = fn

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            self.reset()
        if epoch % self.save_per_n_rounds == 0 and self.best_acc < logs['val_sparse_categorical_accuracy']:
            self.best_acc = logs['val_sparse_categorical_accuracy']
            self.save_model('acc', epoch)
        # if epoch % self.save_per_n_rounds == 0 and self.best_loss > logs['val_loss']:
        #     self.best_loss = logs['val_loss']
        #     self.save_model('loss', epoch)

    @staticmethod
    def delete_model(fn):
        if fn is not None:
            shutil.rmtree(fn)


def parse_args():
    """
    Generate and return arguments
    :return: args
    """
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--model',
                        type=str,
                        default='mlp',
                        choices=['mlp', ''],
                        help='Choices')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default="./checkpoints",
                        help='directory for checkpoint files')
    parser.add_argument('--log_dir',
                        type=str,
                        default="./logs",
                        help='directory for training log')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/UCRArchive_2018',
                        help='Directory to the data folder')
    parser.add_argument('--data_name',
                        type=str,
                        default='Earthquakes',
                        help='Folder name of the dataset to use, set to empty to process all datasets')
    parser.add_argument('--val_portion',
                        type=float,
                        default=0.3,
                        help='validation dataset portion')
    parser.add_argument('--hl_nodes',
                        type=int,
                        default=512,
                        help='Number of node in MLP hidden layers')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='learning rate')
    parser.add_argument('--lr_decay',
                        type=float,
                        default=1.0,
                        help='learning rate decay')
    parser.add_argument('--epochs',
                        type=int,
                        default=300,
                        help='training epochs')
    parser.add_argument('--batch',
                        type=int,
                        default=8,
                        help='batch size')
    parser.add_argument('--workers',
                        type=int,
                        default=6,
                        help='number of workers')
    parser.add_argument('--seed',
                        type=int,
                        default=10,
                        help='seed for dataset shuffling')
    parser.add_argument('--save_log',
                        type=boolean_string,
                        default=True,
                        help='save training log')
    parser.add_argument('--train_patience',
                        type=int,
                        default=10,
                        help='Stop training when val_accuracy does not improve for n consecutive epochs,'
                             'set to 0 to disable early stopping.')
    return parser.parse_args()


def train(args, data_name):
    # load training data
    dataset, unique_y_list = load_data(args.data_dir, data_name, nan_filler=0.0)
    x_train, y_train = dataset['train']['x'], dataset['train']['y']
    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train, dataset_val = partitions_dataset(dataset_train,
                                                    val_portion=args.val_portion,
                                                    shuffle=True,
                                                    seed=args.seed)

    # shuffle training dataset
    train_size = dataset_train.cardinality().numpy()
    dataset_train = dataset_train.shuffle(train_size)
    dataset_train = dataset_train.batch(args.batch)
    dataset_val = dataset_val.batch(args.batch)

    # calculate input and output size
    in_size = x_train.shape[1]
    out_size = len(np.unique(y_train))

    if args.model == 'mlp':
        model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(args.hl_nodes, input_dim=in_size, activation='relu'),
            tf.keras.layers.Dense(args.hl_nodes // 2, activation='relu'),
            tf.keras.layers.Dense(args.hl_nodes // 4, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(out_size),
        ])
    else:
        raise Exception(f'Invalid model type: {args.model}')

    # create learning rate
    if args.lr_decay == 1.0:
        learning_rate = args.lr
    else:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=args.lr,
                                                                       decay_steps=args.epochs//10,
                                                                       decay_rate=args.lr_decay,
                                                                       staircase=True)

    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

    fn_prefix = args.model
    fn_suffix = 'b{}_l{}_h{}'.format(args.batch, args.lr, args.hl_nodes)

    # create checkpoint and log call backs
    callbacks = []
    checkpoint_dir = os.path.join(args.checkpoint_dir, data_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_saver = CheckpointSaver(checkpoint_dir, fn_prefix, fn_suffix, save_per_n_rounds=5)
    callbacks.append(checkpoint_saver)
    if args.train_patience > 0:
        early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                                         patience=args.train_patience)
        callbacks.append(early_stopper)
    if args.save_log:
        logger = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir, histogram_freq=1)
        callbacks.append(logger)

    model.fit(dataset_train,
              validation_data=dataset_val,
              batch_size=args.batch,
              epochs=args.epochs,
              workers=args.workers,
              callbacks=callbacks)
    # perform evaluation on test dataset
    print("\n\nEvaluating test dataset...")

    # Test model accuracy with test dataset
    dataset_test = tf.data.Dataset.from_tensor_slices((dataset['test']['x'], dataset['test']['y']))
    dataset_test = dataset_test.batch(args.batch)
    model.evaluate(dataset_test)


def main():
    args = parse_args()
    if args.data_name:
        train(args, args.data_name)
    else:
        for data_name in tqdm(os.listdir(args.data_dir)):
            if data_name != 'Missing_value_and_variable_length_datasets_adjusted':
                print(f"Start training {data_name}.")
                train(args, data_name)


if __name__ == '__main__':
    main()