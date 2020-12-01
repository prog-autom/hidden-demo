import argparse
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split

from utils import make_dir, DATA_PATH

def load_and_split():
    X, y = datasets.load_iris(return_X_y=True)
    X_pretrain, X_test, y_pretrain, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_dev, y_train, y_dev = train_test_split(X_pretrain, y_pretrain, test_size=0.2)
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def save_data(X_train, X_dev, X_test, y_train, y_dev, y_test):
    def do_pickle(obj, path):
        with open(path, "wb") as fp:
            pickle.dump(obj, fp)

    do_pickle(X_train, f"{DATA_PATH}/X_train.pickle")
    do_pickle(X_dev, f"{DATA_PATH}/X_dev.pickle")
    do_pickle(X_test, f"{DATA_PATH}/X_test.pickle")
    do_pickle(y_train, f"{DATA_PATH}/y_train.pickle")
    do_pickle(y_dev, f"{DATA_PATH}/y_dev.pickle")
    do_pickle(y_test, f"{DATA_PATH}/y_test.pickle")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="")
    run_name = parser.parse_args().run_name

    global DATA_PATH
    DATA_PATH = f"./data/{run_name}"

    make_dir(f"{DATA_PATH}")
    X_train, X_dev, X_test, y_train, y_dev, y_test = load_and_split()
    save_data(X_train, X_dev, X_test, y_train, y_dev, y_test)


if __name__ == "__main__":
    main()
