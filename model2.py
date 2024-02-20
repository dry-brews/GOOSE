#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:22:23 2021

@author: bryan
"""

#  0f 
#  |            
#  |           
#  |--------> MF         
#  |           
#  v           
#  0b 


#Two-state conformational equilibrium with bound and unbound state
#Degradation \propto amount of misfolded protein, fitness \propto amount of protein * dG

import autograd.numpy as anp
kT = 0.001985875 * 298.

def protein_abundance(energy_set, wt_energy_set, r_p = 0.01):
    anp.seterr(over='raise')
    eA_terms = [(energy_set[-1] - energy_set[i])/kT for i in range(len(energy_set)-1)]
    eW_terms = [(wt_energy_set[-1] - wt_energy_set[i])/kT for i in range(len(energy_set)-1)]
    c_norm = max(eA_terms)
    d_norm = max(eW_terms)
    sum_a = anp.sum([anp.exp(term - c_norm) for term in eA_terms])
    sum_w = anp.sum([anp.exp(term - d_norm) for term in eW_terms])
    prot_abund = (sum_a * (anp.exp(-d_norm) + r_p * sum_w)) / (sum_w * (anp.exp(-c_norm) + r_p * sum_a))
    return prot_abund
    
def fitness_prediction(energy_set, wt_energy_set):

    return energy_set[0] * protein_abundance(energy_set, wt_energy_set)


def get_model_info():
    number_of_states = 3
    state_names = ["E_1f", "E_1b", "E_mf"]
    initializations = [0., 0., 0.]
    reference_state = "E_1f"
    quantities_to_calc = ["k_b", "k_u"]
    return (number_of_states, state_names, initializations, reference_state, quantities_to_calc)

def calc_equilibria(target, energy_set):
    if target == "k_b":
        return anp.exp((0. - energy_set[1]) / kT)
    elif target == "k_u":
        return anp.exp(-energy_set[1]/kT) / (1. + anp.exp(-energy_set[0]/kT))