import argparse
import pandas
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import tensorboard_logger

from utils import *


def loadData():
    def loadFromPickle(path):
        with open(path, "rb") as fp:
            result = pickle.load(fp)
        return result

    X_train = loadFromPickle(f"{DATA_PATH}/X_train.pickle")
    X_dev = loadFromPickle(f"{DATA_PATH}/X_dev.pickle")
    y_train = loadFromPickle(f"{DATA_PATH}/y_train.pickle")
    y_dev = loadFromPickle(f"{DATA_PATH}/y_dev.pickle")
    return X_train, X_dev, y_train, y_dev


def loadModel():
    try:
        with open(f"{MODELS_PATH}/model.pickle", "rb") as fp:
            model = pickle.load(fp)
    except FileNotFoundError:
        model = SGDClassifier()
    return model


def saveModel(model):
    with open(f"{MODELS_PATH}/model.pickle", "wb") as fp:
        pickle.dump(model, fp)


def trainModel(model, numOfEpochs, X_train, y_train):
    tensorboard_logger.configure(f"{LOG_PATH}/log_tb/")
    for i in range(1, numOfEpochs):
        model.partial_fit(X_train, y_train, list(set(y_train)))
        y_pred = model.predict(X_train)
        report = classification_report(y_train, y_pred, output_dict=True)
        tensorboard_logger.log_value("f1-score", report["macro avg"]["f1-score"], i)
        if i % 10 == 0:
            with open(f"{LOG_PATH}/log/train_log_{i}", "w") as outp:
                report_raw = classification_report(y_train, y_pred)
                outp.write(report_raw)
    return model


def saveResult(model, X_dev, y_dev):
    y_pred = model.predict(X_dev)
    report = classification_report(y_dev, y_pred, output_dict=True)
    df = pandas.DataFrame(report).transpose()
    df.to_csv(f"{RESULTS_PATH}/report.csv", sep="\t")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()
    run_name = args.run_name
    numOfEpochs = args.n

    global RUN_NAME, DATA_PATH, MODELS_PATH, RESULTS_PATH, LOG_PATH

    RUN_NAME = run_name
    DATA_PATH = "./data"
    MODELS_PATH = f"./models/{run_name}"
    RESULTS_PATH = f"./results/{run_name}"
    LOG_PATH = f"./log/{run_name}"

    make_dir(f"{MODELS_PATH}")
    make_dir(f"{LOG_PATH}/log")
    make_dir(f"{LOG_PATH}/log_tb")
    make_dir(F"{RESULTS_PATH}")

    X_train, X_dev, y_train, y_dev = loadData()
    model = loadModel()
    model = trainModel(model, numOfEpochs, X_train, y_train)
    saveModel(model)
    saveResult(model, X_dev, y_dev)


if __name__ == "__main__":
    main()
