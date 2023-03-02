# Welcome to the hidden loops demo!

This repo contains the source code accompanying the paper on hidden loops 
named "Hidden Feedback Loops in Machine Learning Systems: A Simulation Model and Preliminary Results"

In this concept paper, we explore some of the aspects of quality of continuous learning artificial
intelligence systems as they interact with and influence their environment. We study an important problem
of implicit feedback loops that occurs in recommendation systems, web bulletins and price estimation
systems. We demonstrate how feedback loops intervene with user behavior on an exemplary housing
prices prediction system. Based on a preliminary model, we highlight sufficient existence conditions when
such feedback loops arise and discuss possible solution approaches.

## Problem statement

We running the experiment as show at the figure below 

<img src=".img/experiment-setup.png" alt="experiment setup" width="700"/>

## How to run

There are two experiments included in this repo.

 - single model experiment that demonstrates how housing prices prediction can be solved 
 - hidden loops experiment shows the feedback loop effect as dscribed in the paper

In order to run any experiment, get the repo and install its requirements:

```bash
$ git clone https://github.com/prog-autom/hidden-demo.git
$ cd ./hidden-demo && pip install -r requirements.txt
```

For single model experiment run

```bash
$ python ./src/main.py single-model \
         --folder ./results/single-model \
         --model_params "{\"gbr_model\": {\"n_estimators\": 150, \"max_depth\": 3, \"criterion\": \"mae\", \"loss\": \"huber\"}, \"ridge_model\": {}}" \
         --params "{\"train_size\": 0.3}" \
         --random_seed 42 
```

For hidden loops you need

```bash
$ python ./src/main.py hidden-loop \
        --folder ./results/hidden-loop \
        --model_params "{\"gbr_model\": {\"n_estimators\": 150, \"max_depth\": 3, \"criterion\": \"mae\", \"loss\": \"huber\"}, \"ridge_model\": {}}" \
        --params "{\"adherence\": 0.2, \"usage\": 1.0, \"step\": 10}" \
        --random_seed 42
```
Resulting figures will be placed in the corresponding ``./results`` folder. 


## How to run with mldev 

Running the same experiment with [mldev](https://gitlab.com/mlrep/mldev) involves the following steps.

Install the ``mldev`` by executing

```bash
$ git clone https://github.com/prog-autom/hidden-demo.git
$ curl https://gitlab.com/mlrep/mldev/-/raw/develop/install_mldev.sh -o install_mldev.sh
$ chmod +x install_mldev.sh
$ yes n | install_mldev.sh
``` 
Then initialize the experiment, this will install required dependencies

```bash
$ mldev --config ./hidden-demo/.mldev/config.yaml init --no-commit -r ./hidden-demo
```

Detailed description of the experiment can be found in ``./experiment.yml``. See docs for ``mldev`` for details.

And now, run the experiment

```bash
$ cd ./hidden-demo && mldev --config ./hidden-demo/.mldev/config.yaml run --no-commit -f experiment.yml pipeline
```

Results will be placed into ``./results`` folder as well.

## Complete experiment with mldev

There is a script ``./run_experiment.sh`` that runs the experiment
for a grid of parameters, usage from 0.1 to 0.9, adherenec from 0.1 to 0.9 
and step size 10 or 20.

The script relies on mldev to run trials for a fixed set of parameters.

## Source code

Source code can be found in ``./src`` folder. The [main.py](./src/main.py) file contains glue code to run experiments.
The [experiment.py](./src/experiment.py) contains experiment implementation and utility procedures.

## Citing

If you find results useful, please cite the paper and the reproducibility tool as following

```bibtex
@inproceedings{Khritankov2021Hidden,
    title = "Hidden feedback loops in machine learning systems: a simulation model and preliminary results",
    author = "Khritankov, Anton",
    booktitle = "Proceedings of the 13th International Conference, SWQD 2021, Vienna, Austria, January 19–21, 2021",
    year = "2021",
    pages={54--65},
    volume = "404",
    series = "Software Quality: Future Perspectives on Software Engineering Quality",
    publisher = "Springer International Publishing"
}

@InProceedings{10.1007/978-3-031-12285-9_1,
 author="Khritankov, Anton and Pershin, Nikita and Ukhov, Nikita and Ukhov, Artem",
 editor="Pozanenko, Alexei and Stupnikov, Sergey and Thalheim, Bernhard and Mendez, Eva and Kiselyova, Nadezhda",
 title="MLDev: Data Science Experiment Automation and Reproducibility Software",
 booktitle="Data Analytics and Management in Data Intensive Domains",
 year="2022",
 publisher="Springer International Publishing",
 address="Cham",
 pages="3--18",
 isbn="978-3-031-12285-9"
}

```

There is an updated version of the paper at [arXiv:2101.05673](https://arxiv.org/abs/2101.05673)

## License

The code is licensed under MIT license, see [LICENSE](LICENSE)



