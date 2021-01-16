#!/bin/bash

trap 'exit' SIGINT SIGTERM SIGHUP SIGQUIT

for step in 10 20
do
    for usage in 0.{1..9..2}
    do
        for adherence in 0.{1..9..2}
        do
          export step &&
          export usage &&
          export adherence &&
          mldev --config .mldev/config.yaml run -f ./experiment.yml --no-commit run_experiment
        done
    done
done