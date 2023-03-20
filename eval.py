import argparse
from utils import *
select_gpu(0, GB(4))


def parse_args():
    """
    Generate and return arguments
    :return: args
    """
    parser = argparse.ArgumentParser(description='Model evaluation')
    parser.add_argument('--model_checkpoint',
                        type=str,
                        default="mlp_loss0.0004_e95_b16_l0.0001_h512",
                        help='Folder name of the model checkpoint')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default="./checkpoints",
                        help='directory for checkpoint files')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/UCRArchive_2018',
                        help='Directory to the data folder')
    parser.add_argument('--data_name',
                        type=str,
                        default='Earthquakes',
                        help='Folder name of the dataset to use')
    parser.add_argument('--val_portion',
                        type=float,
                        default=0.3,
                        help='validation dataset portion')
    parser.add_argument('--batch',
                        type=int,
                        default=16,
                        help='batch size')
    parser.add_argument('--workers',
                        type=int,
                        default=6,
                        help='number of workers')
    parser.add_argument('--seed',
                        type=int,
                        default=10,
                        help='seed for dataset shuffling')
    return parser.parse_args()


def main():
    # create arguments
    args = parse_args()

    # load and prepare test dataset
    dataset, unique_y_list = load_data(args.data_dir, args.data_name)
    dataset_test = tf.data.Dataset.from_tensor_slices((dataset['test']['x'], dataset['test']['y']))
    dataset_test = dataset_test.batch(args.batch)

    # load model from checkpoint and make predictions
    model = tf.keras.models.load_model(os.path.join(args.checkpoint_dir, args.data_name, args.model_checkpoint))

    # tf.summary.trace_on(graph=True, profiler=True)
    _, acc = model.evaluate(dataset_test)
    print(f'Model accuracy = {acc}')
    # with writer.as_default():
    #     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)


if __name__ == '__main__':
    main()
