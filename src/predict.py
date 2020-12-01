import argparse
import pickle
import pandas
from sklearn.metrics import classification_report

from utils import MODELS_PATH, make_dir,RESULTS_PATH,LOG_PATH,RUN_NAME,DATA_PATH

def load_data():
    def unpickle(path):
        with open(path, "rb") as fp:
            result = pickle.load(fp)
        return result
    
    X_train = unpickle(f"{DATA_PATH}/X_train.pickle")
    X_dev = unpickle(f"{DATA_PATH}/X_dev.pickle")
    X_test = unpickle(f"{DATA_PATH}/X_test.pickle")
    y_train = unpickle(f"{DATA_PATH}/y_train.pickle")
    y_dev = unpickle(f"{DATA_PATH}/y_dev.pickle")
    y_test = unpickle(f"{DATA_PATH}/y_test.pickle")
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def load_model():
    with open(f"{MODELS_PATH}/model.pickle", "rb") as fp:
        model = pickle.load(fp)
    return model


def predict_and_save(model, X, y, path):
    y_pred = model.predict(X)
    report = classification_report(y, y_pred, output_dict=True)
    df = pandas.DataFrame(report).transpose()
    df.to_csv(path, sep="\t")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="default")
    run_name = parser.parse_args().run_name

    global RUN_NAME, DATA_PATH, MODELS_PATH, RESULTS_PATH, LOG_PATH

    RUN_NAME = run_name
    DATA_PATH = "./data"
    MODELS_PATH = f"./models/{run_name}"
    RESULTS_PATH = f"./results/{run_name}"
    LOG_PATH = f"./log/{run_name}"

    X_train, X_dev, X_test, y_train, y_dev, y_test = load_data()
    model = load_model()
    predict_and_save(model, X_train, y_train, f"{RESULTS_PATH}/train_report.csv")
    predict_and_save(model, X_dev, y_dev, f"{RESULTS_PATH}/dev_report.csv")
    predict_and_save(model, X_test, y_test, f"{RESULTS_PATH}/test_report.csv")


if __name__ == "__main__":
    main()