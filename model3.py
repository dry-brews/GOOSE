#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:22:23 2021

@author: bryan
"""

#  0f ---------> 1f
#  |            |
#  |            |
#  |            |
#  |            |
#  v            v
#  0b ---------> 1b


#Two-state conformational equilibrium with bound and unbound state
#No consideration of protein abundance or misfolded states

import autograd.numpy as anp
kT = 0.001985875 * 298.

def fitness_prediction(energy_set, wt_energy_set):
    #energy_set = [{E_0f}, E_1f, E_0b, E_1b]
    return -kT * (anp.logaddexp(0, (energy_set[1] - energy_set[2])/kT) -
                  anp.logaddexp(0, (0. - energy_set[0])/kT)) - \
                          0. + energy_set[1]

def get_model_info():
    number_of_states = 4
    state_names = ["E_1f", "E_2f", "E_1b", "E_2b"]
    initializations = [0., 0., 0., 0.]
    reference_state = "E_1f"
    quantities_to_calc = ["k_b", "k_c", "alpha"]
    return (number_of_states, state_names, initializations, reference_state, quantities_to_calc)

def calc_equilibria(target, energy_set):
    if target == "k_b":
        return anp.exp((0. - energy_set[1]) / kT)
    elif target == "k_c":
        return anp.exp((0. - energy_set[0]) / kT)
    elif target == "alpha":
        return anp.exp((energy_set[0] + energy_set[1] - energy_set[2]) / kT)