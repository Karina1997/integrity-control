import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from pathlib import Path

MODEL_FILENAME = 'additional_data/model.pt'
RESULT_FILENAME = 'additional_data/result.csv'


def train(data):
    train_y = data.iloc[:, 0]
    train_x = data.drop(0, axis=1)
    clf = LogisticRegression(random_state=42, solver='liblinear', penalty='l1', max_iter=10000).fit(train_x, train_y)
    pickle.dump(clf, open(MODEL_FILENAME, 'wb'))
    print(str(Path().absolute()) + "/" + MODEL_FILENAME)


def test(data, pretrained_weights_path):
    loaded_model = pickle.load(open(pretrained_weights_path, 'rb'))
    y = loaded_model.predict(data)
    df = pd.DataFrame(y)
    df.to_csv(RESULT_FILENAME, index=False, header=False)
    print(str(Path().absolute()) + "/" + RESULT_FILENAME)


def control(mode, csv_file_path, pretrained_weights=None):
    data = pd.read_csv(csv_file_path, header=None)
    if mode == "train":
        train(data)
    else:
        test(data, pretrained_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="mode should be train or test", type=str)
    parser.add_argument("path", help="path to csv file should be specified", type=str)
    parser.add_argument("pretrained_weights", help="should be specified for test mode", nargs='?', default=None)
    args = parser.parse_args()
    if args.mode not in ["test", "train"]:
        print("Wrong mode argument value")
        raise Exception
    if args.mode == "test":
        if args.pretrained_weights is None:
            print("Pretrained weights should be specified for test mode")
            raise Exception
    control(args.mode, args.path, args.pretrained_weights)
