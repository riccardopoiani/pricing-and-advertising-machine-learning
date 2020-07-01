# Data Intelligence Application - 2019 
Group members:
- Alessandrelli Luca
- Gargano Jacopo
- Poiani Riccardo
- Zhou Tang-Tang

Chosen topic: Pricing + Advertising

## Instructions
In requirements it is possible to find the necessary dependencies to run the project. <br> <br>
All experiments are in folder /experiment. Each of them generates files (that can be read, for instance, 
with pickle) containing the results of the experiment and a txt file containing details on the setting
of the experiment. <br>
To run an experiment, one can simply write "python exp_name.py -n_runs x -s 1": this will launch x runs 
for a given experiment and will save results on file (notice that it is necessary to run it from the folder /experiment).
Additional parameters can be set to control the environment parameters and algorithm hyper-parameters (--help 
to see details on each of this additional possibility). <br> <br>

## Environment
Functions generating data (i.e. the environment) are controlled in /resources. 
In each file, conversion rate probabilities (CRP) stochastic functions and number of clicks 
stochastic functions are defined for different users classes, as well as possibly different
breakpoints to simulate a non-stationary environment with abrupt changes. 
To change the environment while running experiments just add "--scenario_name file_name".


