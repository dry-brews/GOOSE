GOOSE: Gradient-descent Optimization Of State Energies

This repository contains python code for interpreting double mutational data of proteins where mutations are expected to additively affect underlying discrete states of the protein.
The script can handle arbitrary, but explicitly-defined, models that link a number of states to protein function. Four such models are provided:
Model 1 assumes two states, bound and unbound
Model 2 assumes three states, folded+unbound, folded+bound, and unfolded+unbound(+destined for degradatation)
Model 3 assumes four states with two conformations, conf1+unbound, conf2+unbound, conf1+bound, conf2+unbound,
Model 4 assumes five states with two conformations, conf1+unbound, conf2+unbound, conf1+bound, conf2+unbound, and unfolded+unbound(+destined for degradatation)

Other models may be provided by the user by creating a model.py file that contains a list of states and a function that predicts fitness given a set of values for those states.

Dependencies and requirements:
Python3
argparse, autograd, random, scipy, multiprocessing, time, uuid,

Recommended Usage:
Create a shell script (modelled off the example script) whose variable point to the dataset, model, and python scripts that you want to use.
The script will automatically divide your data into a training and test set, fit the energies, and return the predicted scores of double mutations for your training and test sets.
While fitting energies, it is important to provide an accurate estimate of the wildtype fitness using the -w flag. Furthermore, the script benefits from multithreading, use the -n flag to specify how many threads you want to dedicate.
