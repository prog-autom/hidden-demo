import os

RUN_NAME = None
RESULTS_PATH = None
MODELS_PATH = None
DATA_PATH = None
LOG_PATH = None


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)