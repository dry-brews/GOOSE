#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:38:10 2021

@author: bryan
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 15:22:23 2021

@author: bryan
"""

#  0f
#  |            
#  |            
#  |            
#  |            
#  v            
#  0b


#No conformational equilibria, just bound and unbound state
#No consideration of protein abundance or misfolded states

import autograd.numpy as anp
kT = 0.001985875 * 298.

def fitness_prediction(energy_set, wt_energy_set):
    #energy_set = [{E_0f}, E_1f, E_0b, E_1b]
    return energy_set[0]

def get_model_info():
    number_of_states = 2
    state_names = ["E_1f", "E_1b"]
    initializations = [0., 0.]
    reference_state = "E_1f"
    quantities_to_calc = ["k_b"]
    return (number_of_states, state_names, initializations, reference_state, quantities_to_calc)

def calc_equilibria(target, energy_set):
    if target == "k_b":
        return anp.exp((0. - energy_set[0]) / kT)