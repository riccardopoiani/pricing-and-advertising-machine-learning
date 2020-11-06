# Machine learning techniques for pricing and advertising

## Content highlights
Implementation and experiments on machine learning techinques for pricing and advertising are available within this repository. "dia_report.pdf" contains details on the purpose and the content of the repository; together with results obtained in simulated domains. Here, we provide a list of what the repository contains:
- Multi-armed bandit techniques for pricing: Thomposon sampling, EXP-3, UCB-1, and UCB extensions that exploit reasonable hypothesis regarding the pricing problem
- Non-stationary multi-armed bandit techinques (both change point detection and sliding windows) for non-stationary environments 
- Contextual multi-armed bandit technique: greedy-context generation algorithm
- Dynamic-programming for campaing budget optimization algorithm together with Gaussian Processes estimators, to solve advertising problems
- Approaches for solving a contextual joint problem of pricing with advertising

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


