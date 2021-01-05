import argparse
import json
import os

from experiment import *

parser = None
folder = "./results"


def create_model(model_params):
    if 'gbr_model' in model_params:
        model_name = 'gbr'
        model = lambda : gbr_model(**model_params['gbr_model'])
        print(f"Using model: {model_name}")
        yield model, model_name
    if 'ridge_model' in model_params:
        model_name = 'ridge'
        model = lambda : ridge_model(**model_params['ridge_model'])
        print(f"Using model: {model_name}")
        yield model, model_name


def single_model(model_params, params):
    print(f"Running single-model experiment")
    X, y, _ = get_boston_dataset()
    for model, model_name in create_model(model_params):
        single_model_experiment(X, y, model, model_name=f"{folder}/{model_name}", **params)


def hidden_loop(model_params, params):
    print(f"Running hidden-loop experiment")
    X, y, _ = get_boston_dataset()
    for model, model_name in create_model(model_params):
        hle = HiddenLoopExperiment(X, y, model, model_name)
        prepare_params = {k: params[k] for k in params.keys() & {'train_size'}}
        hle.prepare_data(**prepare_params)

        loop_params = {k: params[k] for k in params.keys() & {'seed', 'adherence', 'usage', 'step'}}
        hle.hidden_loop_experiment(**loop_params)

        hle.plot_results(folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", type=str, help="Kind of experiment: single-model or hidden-loop")
    parser.add_argument("--params", type=str, help="A json string with experiment parameters")
    parser.add_argument("--model_params", type=str, help="A json string with model name and parameters")
    parser.add_argument("--folder", type=str, help="Save results to this folder")
    args = parser.parse_args()
    model_str = args.model_params
    params_str = args.params
    kind = args.kind
    folder = args.folder
    os.makedirs(folder, exist_ok=True)

    model_dict = json.loads(model_str)
    params_dict = json.loads(params_str)

    if kind == "single-model":
        single_model(model_dict, params_dict)
    elif kind == "hidden-loop":
        hidden_loop(model_dict, params_dict)
    else:
        parser.error("Unknown experiment kind: " + kind)