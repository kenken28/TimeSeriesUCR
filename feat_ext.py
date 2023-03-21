import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import extract_features
from tsfresh import select_features
from utils import *


def parse_args():
    """
    Generate and return arguments
    :return: args
    """
    parser = argparse.ArgumentParser(description='Model trainer')
    parser.add_argument('--data_name',
                        type=str,
                        default='',
                        help='Folder name of the dataset to use, set to empty to process all datasets')
    parser.add_argument('--data_dir',
                        type=str,
                        default='./data/UCRArchive_2018',
                        help='Directory to the data folder')
    parser.add_argument('--result_dir',
                        type=str,
                        default="./results",
                        help='directory for checkpoint files')
    return parser.parse_args()


def get_tsfresh_data(data):
    x_tsfresh = []
    y_tsfresh = dict()
    for idx, (x, y) in enumerate(zip(data['x'], data['y'])):
        sample_dict = {
            'id':    [idx] * len(x),
            'time':  list(range(len(x))),
            'value': x.tolist(),
        }
        x_tsfresh.append(pd.DataFrame(sample_dict))
        y_tsfresh[idx] = y
    x_tsfresh = pd.concat(x_tsfresh, axis=0)
    y_tsfresh = pd.Series(y_tsfresh)

    return x_tsfresh, y_tsfresh


def analyse_features(args, data_name):
    data_dict, _ = load_data(args.data_dir, data_name)
    x_train, y_train = get_tsfresh_data(data_dict['train'])
    x_test, y_test = get_tsfresh_data(data_dict['test'])

    x_train_feat = extract_features(x_train, column_id='id', column_sort='time', impute_function=impute)
    x_train_feat_filtered = select_features(x_train_feat, y_train)
    if not x_train_feat_filtered.empty:
        x_train_feat = x_train_feat_filtered

    rand_forest = RandomForestClassifier()
    rand_forest.fit(x_train_feat, y_train)

    x_test_feat = extract_features(x_test, column_id='id', column_sort='time', impute_function=impute)
    x_test_feat = x_test_feat.loc[:, x_train_feat.columns]
    pred = rand_forest.predict(x_test_feat)

    acc = accuracy_score(y_test.values, pred)
    print(f'{data_name} accuracy = {acc}')
    record_test_accuracy(args.result_dir, 'FEAT_RandForest', data_name, acc)


def main():
    args = parse_args()
    if args.data_name:
        analyse_features(args, args.data_name)
    else:
        data_name_list = os.listdir(args.data_dir)
        for i, data_name in enumerate(data_name_list):
            if data_name != 'Missing_value_and_variable_length_datasets_adjusted':
                print(f'[{i}/{len(data_name_list)}] Start training {data_name}.')
                analyse_features(args, data_name)


if __name__ == '__main__':
    main()
