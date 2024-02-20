#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2 12:06:21 2021

@author: bryan
"""

'''
Goals for the current update:
    1. Explicit epoch framework to make sure all the data gets run through at an equal pace (epochs)
    2. Explicitly normalize the gradient per number of observations
    -check
    3. Add option to change recording interval

'''
#Set-up########################################################################
#Before getting to main(), let's import our protein model and set up some
#maths behind-the-scenes
#main() will get set up using argparse at bottom

#units = kcal / mol
kT = 0.001985875 * 298.



from protein_model import fitness_prediction, get_model_info, calc_equilibria
#Ok, lets read in the model info from protein model############################
num_states, state_names, inits, ref, quants = get_model_info()

#This function applies the basic fitness_prediction to all scores
def predict_scores(params, scores, mutID_dict):
    preds = []
    params = anp.reshape(params, (-1,(num_states-1)))
    for s in scores:
        mutA_energies = params[mutID_dict[s[0]]]
        mutB_energies = params[mutID_dict[s[1]]]
        wt_energies = params[0]
        mutAB_energies = [mutA_energies[i] + mutB_energies[i] - wt_energies[i] for i in range(len(wt_energies))]
        #print(mutID_dict[s[0]], mutID_dict[s[1]], mutA_energies, mutB_energies, wt_energies, mutAB_energies)
        preds.append(fitness_prediction(mutAB_energies, wt_energies))
    return preds

#This function initializes the parameters
def initialize_params(inits = inits, number_of_muts = 100):
    one_energy_set = inits
    del one_energy_set[state_names.index(ref)]
    params = anp.array([one_energy_set for i in range(number_of_muts)], dtype=anp.float64)
    return params

#Main##########################################################################
def main(input_file, output_file, offset, reload_file):
    
    #Read in scores from file. Scores are normalized so wt=0, values are dG in kcal / mol
    #mutID_dict takes a mutation and tells you where the parameters for that
    #variant are located in the matrix of parameters
    scores, mutID_dict, singles = read_doubles_file(input_file)
    
    #Add singles scores to training set. 100% of singles scores get added, even
    #though the training set is not 100% of the doubles scores. Should fix
    scores = add_singles_scores(scores, singles)

    #create output file and remove any existing files with same name
    with open(output_file,'w+') as file_out:
        file_out.write('\t'.join([
                "pos", "aa", '\t'.join(quants),
                '\t'.join(state_names), "dG_pred",
                "dG_meas", 'iter']) + '\n')

    free_params = initialize_params(inits=inits, number_of_muts = len(mutID_dict))
    sys.stderr.write("loading in estimates from %s ...\n" % reload_file)
    iter_start, free_params = load_params(reload_file, free_params, mutID_dict) 

    
    free_params = anp.reshape(free_params, -1)

    wt_energies = free_params[:num_states-1]

    predictions = predict_scores(free_params, scores, mutID_dict)
    
    i = 0
    with open(output_file,'w+') as file_out:
        file_out.write("pos1\taa1\tpos2\taa2\tdG_pred\tdG_meas\n")
        
        for s in scores:
            #pos1 = s[0].split("_")[0]
            #pos2 = s[1].split("_")[0]
            #aa1 = s[0].split("_")[1]
            #aa2 = s[1].split("_")[1]
            #dG_pred = predictions[i]
            #dG_meas = scores[s] + offset
            file_out.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (s[0].split("_")[0],
                                                         s[0].split("_")[1],
                                                         s[1].split("_")[0],
                                                         s[1].split("_")[1],
                                                         predictions[i],
                                                         scores[s] + offset
                                                         ))

            i +=1

def load_params(prev_output_file, params, mutID_dict):
    with open(prev_output_file,'r') as tsv_in:   
        lc = 0
        for line in tsv_in:
            lc +=1
            if lc ==1:
                header = line.strip().split('\t')
                energy_inds = [header.index(e) for e in state_names]
                pos_ind = header.index("pos")
                aa_ind = header.index("aa")
                iter_ind = header.index("iter")
            else:
                row = line.strip().split('\t')
                energies = [row[e] for e in energy_inds]
                del energies[state_names.index(ref)]
                mut_ID = row[pos_ind] + "_" + row[aa_ind]
                try:
                    params[mutID_dict[mut_ID]] = energies
                except IndexError:
                    sys.stderr.write("Failed to read existing data file with mutID=%s, exiting" % mut_ID)
                    sys.exit()
                curr_iter = int(row[iter_ind])
    return (curr_iter, params)

def read_doubles_file(filename):
    doubles_scores = {}
    singles_scores = {}
    positions = []
    mutIDs = {"0_0": 0}
    with open(filename, 'r') as tsv_in:
        lc = 0
        for line in tsv_in:
            lc +=1
            if lc == 1:
                header = line.strip().split('\t')
                indices = {}
                for col in header:
                    indices[col] = header.index(col)
            else:
                row = line.strip().split('\t')
                if int(row[indices["pos1"]]) < int(row[indices["pos2"]]):
                    mut1_UID = '_'.join([row[indices["pos1"]], row[indices["aa1"]]])
                    mut2_UID = '_'.join([row[indices["pos2"]], row[indices["aa2"]]])
                    try:
                        singles_scores[mut1_UID] = float(row[indices["dG_1"]])
                    except ValueError:
                        singles_scores[mut1_UID] = "NA"
                    try:
                        singles_scores[mut2_UID] = float(row[indices["dG_2"]])
                    except ValueError:
                        singles_scores[mut2_UID] = "NA"
                    doubles_scores[(mut1_UID, mut2_UID)] = float(row[indices["dG_double"]])
                else:
                    sys.stderr.write("Error: pos1 >= pos2 in row %s, exiting..." % lc)
                    sys.exit()
                if row[indices["pos1"]] not in positions:
                    positions.append(row[indices["pos1"]])
                if row[indices["pos2"]] not in positions:
                    positions.append(row[indices["pos2"]])
                if mut1_UID not in mutIDs:
                    mutIDs[mut1_UID] = len(mutIDs)
                if mut2_UID not in mutIDs:
                    mutIDs[mut2_UID] = len(mutIDs)
    return (doubles_scores, mutIDs, singles_scores)

def add_singles_scores(scores, singles):
    for s in singles:
        if singles[s] == "NA":
            continue
        scores[(("0_0") , s)] = singles[s]
    return scores


if __name__ == "__main__":

    import argparse
    import sys
    import autograd.numpy as anp
    import os
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i",
                  "--input",
                  action = 'store',
                  dest="input_scores",
                  help="input scores file (as a tsv with pos1, pos2, aa1, aa2, deltaG_Double in the header)"
                  )
    parser.add_argument("-o",
                  "--output",
                  action = 'store',
                  dest="output_file",
                  default="temp.tsv",
                  help="Output file to write data to (writes estimates every iteration)"
                  )
    parser.add_argument("-w",
                  "--offset",
                  action = 'store',
                  dest="offset",
                  type=float,
                  default="-8.23",
                  help="If necessary, shift all measured scores by a set amount (e.g., to normalize to wt)"
                  )
    parser.add_argument("-r",
                  "--reload-file",
                  action = 'store',
                  dest="reload_file",
                  type=str,
                  default="NA",
                  help="Load in a file from previous optimization to continue optimizing"
                  )


    options = parser.parse_args()

    main(options.input_scores, options.output_file, options.offset, options.reload_file)

