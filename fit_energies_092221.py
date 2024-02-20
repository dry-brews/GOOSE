#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on July 2 12:06:21 2021

@author: bryan
"""

#Set-up########################################################################
#Before getting to main(), let's import our protein model and set up some
#maths behind-the-scenes
#main() will get set up using argparse at bottom

#units = kcal / mol
kT = 0.001985875 * 298.

from autograd import grad
import uuid
import sys
import random

from protein_model import fitness_prediction, get_model_info, calc_equilibria
#Ok, lets read in the model info from protein model############################
num_states, state_names, inits, ref, quants = get_model_info()

#This function creates a global-level clone of functions that are locally defined
#It is necessary to allow autograd to access locally defined functions
def globalize(func):
  def result(*args, **kwargs):
    return func(*args, **kwargs)
  result.__name__ = result.__qualname__ = uuid.uuid4().hex
  setattr(sys.modules[result.__module__], result.__name__, result)
  return result

#This function applies the basic fitness_prediction to all scores
def predict_scores(params, scores, mutID_dict, num_threads, wt_energies, fix_wt = False):
    preds = []
    params = anp.reshape(params, (-1,(num_states-1)))
    if fix_wt == True:
        wt_energies = params[0]
    for s in scores:
        mutA_energies = params[mutID_dict[s[0]]]
        mutB_energies = params[mutID_dict[s[1]]]
        mutAB_energies = [mutA_energies[i] + mutB_energies[i] - wt_energies[i] for i in range(len(wt_energies))]
        #print(mutID_dict[s[0]], mutID_dict[s[1]], mutA_energies, mutB_energies, wt_energies, mutAB_energies)
        preds.append(fitness_prediction(mutAB_energies, wt_energies))
    return preds

#This is the loss function, which calls on predict_scores()
def loss_func(params, measures, scores, mutID_dict, num_threads, wt_energies, fix_wt):       
    params = anp.reshape(params, (-1,num_states-1))
    preds = predict_scores(params, scores, mutID_dict, num_threads, wt_energies)
    loss = anp.sum([(measures[i] - preds[i])**2 for i in range(len(preds))])
    return loss

def mt_loss_func(params, measures, scores, mutID_dict, num_threads, wt_energies, chunk_size, fix_wt):       
    if chunk_size == 0:
        chunk_size = len(scores) // num_threads
    scores_list = list(scores.keys())
    all_scores = [(scores_list[i], measures[i]) for i in range(len(measures))]
    random.shuffle(all_scores)
    sub_losses = []
    with Pool(processes=num_threads) as pool:
        threadings = []
        for i in range(num_threads):
            scores_subset = [all_scores[i][0] for i in range(chunk_size)]
            measures_subset = [all_scores[i][1] for i in range(chunk_size)]
            all_scores = all_scores[chunk_size:]
            threadings.append(pool.apply_async(loss_func, [params, measures_subset, scores_subset, mutID_dict, num_threads, wt_energies, fix_wt]))
        for t in threadings:
            sub_losses.append(t.get(timeout=10))
    loss = anp.sum(sub_losses)
    return loss

#This is the autodiff gradient of the loss function
grad_loss = globalize(grad(loss_func))

#This function initializes the parameters
def initialize_params(inits = inits, number_of_muts = 100):
    one_energy_set = inits
    del one_energy_set[state_names.index(ref)]
    params = anp.array([one_energy_set for i in range(number_of_muts)], dtype=anp.float64)
    return params

#This function does a fast line search to approximate the optimal step size using a subset of the data
def fast_line_search(params, d_i, measures, scores, mutID_dict, num_threads, wt_energies, chunk_size, offset, beta_course = 0.1, beta_fine = 0.9, alpha = 0.1):
    #alpha here is acceptable error in determining step size, not step size itself
    number_of_muts = len(params) * 10
    loss_inflation = len(measures) / number_of_muts
    score_subset = random.sample(scores.items(), number_of_muts)
    score_subset = {score_subset[i][0]: score_subset[i][1] for i in range(number_of_muts)}
    measure_subset = [score_subset[k]+offset for k in score_subset]

    n = 1.
    #pre-calculate f(x^k)
    fxk = loss_func(params, measure_subset, score_subset, mutID_dict, num_threads, wt_energies, fix_wt = False) *loss_inflation
    #pre-calculate norm
    d_i_norm = anp.linalg.norm(-d_i)**2 * alpha
    
    #First, get in the ballpark with a course search
    while True:
        try:
            check_func = loss_func(params - n * -d_i, measure_subset, score_subset, mutID_dict, num_threads, wt_energies, fix_wt = False)*loss_inflation > ( fxk - n * d_i_norm )
        except FloatingPointError:
            n *= beta_course
            #print("Hit an overflow during step size calculation. Don't worry about it.")
            continue
        if check_func == True:
            n *= beta_course
            continue
        break
    
    #Back up one step
    n = n / beta_course

    #Then do a much finer search    
    while loss_func(params - n * -d_i, measure_subset, score_subset, mutID_dict, num_threads, wt_energies, fix_wt = False)*loss_inflation > ( fxk - n * d_i_norm ):
        n *= beta_fine
    return n  
    
def polak_ribiere_step(gradient_k1, gradient_k):
    if gradient_k is None:
        return -gradient_k1
    else:
        return anp.matmul(gradient_k1.T, (gradient_k1-gradient_k)) / anp.matmul(gradient_k.T, gradient_k) * gradient_k - gradient_k1
    

def mt_gradient(params, scores, mutID_dict, dG_wt, num_threads, wt_energies, chunk_size, fix_wt = False):
    if chunk_size ==0:
        chunk_size = len(scores) // num_threads
    all_scores = list(scores.keys())
    random.shuffle(all_scores)
    gradients = []
    with Pool(processes=num_threads) as pool:
        threadings = []
        for i in range(len(scores)//chunk_size):
            scores_subset = all_scores[:chunk_size]
            all_scores = all_scores[chunk_size:]
            measures = anp.array([scores[s] + dG_wt for s in scores_subset])
            threadings.append(pool.apply_async(grad_loss, [params, measures, scores_subset, mutID_dict, num_threads, wt_energies, fix_wt]))
        gradients = [t.get(timeout=100) for t in threadings]
    full_grad = anp.sum(gradients, axis=0)
    return full_grad

def find_normalization_vector(params, scores, mutID_dict, fix_wt = False):
    occurences = [0 for i in range(len(params))]
    for s in scores:
        for i in range(num_states-1):
            occurences[mutID_dict[s[0]]*(num_states-1) + i] +=1
            occurences[mutID_dict[s[1]]*(num_states-1) + i] +=1
            if fix_wt == True:
                occurences[i] +=1
    norm_constant = len(mutID_dict) - (num_states - 1)
    norm_vector = [norm_constant / occ for occ in occurences]
    return norm_vector

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

def write_ouput_iter(output_file, params, mutID_dict, singles, dG_wt, i, num_threads, wt_energies):
    with open(output_file,'a') as file_out:
        #Write wt first (aa=0, pos=0)
        write_energies = '\t'.join(["0.0"] + [str(e) for e in wt_energies])
        write_equilibria = '\t'.join([str(calc_equilibria(target = t, energy_set = wt_energies)) for t in quants])
        
        file_out.write('\t'.join([
                "0",
                "0",
                write_equilibria,
                write_energies,
                str(round(fitness_prediction(wt_energies, wt_energies),3)),
                str(round(dG_wt,3)),
                str(i)]) + '\n')
    
        #Write the rest of the muts
        for k in sorted(mutID_dict.keys()):
            if k == "0_0":
                continue
            mut_loc = mutID_dict[k]
            mut_energies = params[(num_states-1)*mut_loc:(num_states-1)*mut_loc+(num_states-1)]
            write_energies = '\t'.join(["0.0"] + [str(e) for e in mut_energies])
            write_equilibria = '\t'.join([str(calc_equilibria(target = t, energy_set = mut_energies)) for t in quants])
            try:
                empir_score = str(round(singles[k]+dG_wt,3))
            except TypeError:
                empir_score = "NA"
            file_out.write('\t'.join([
                k.split('_')[0],
                k.split('_')[1],
                write_equilibria,
                write_energies,
                str(round(fitness_prediction(mut_energies, wt_energies),3)),
                empir_score,
                str(i)]) + '\n')
        file_out.close()
    
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
    sys.stderr.write("Read scores for file: %s\n" % filename)
    sys.stderr.write("Mutations occured at %s positions\n" % len(positions))
    possible_muts = int(19 * 19 * len(positions) * (len(positions) - 1) / 2)
    sys.stderr.write("Of %s possible mutations, %s were read (%.2f%%)\n" % (
            possible_muts, len(doubles_scores), float(len(doubles_scores))/ float(possible_muts) * 100.))
    return (doubles_scores, mutIDs, singles_scores)

def add_singles_scores(scores, singles):
    for s in singles:
        if singles[s] == "NA":
            continue
        scores[(("0_0") , s)] = singles[s]
    return scores
#Main##########################################################################
def main(input_file, output_file, offset, tolerance, max_iter, num_threads, reload_file, chunk_size, recording_interval):
    
    #Read in scores from file. Scores are normalized so wt=0, values are dG in kcal / mol
    #mutID_dict takes a mutation and tells you where the parameters for that
    #variant are located in the matrix of parameters
    scores, mutID_dict, singles = read_doubles_file(input_file)
    
    #Add singles scores to training set. 100% of singles scores get added, even
    #though the training set is not 100% of the doubles scores. Should fix
    scores = add_singles_scores(scores, singles)

    #Initialize parameters based on prior expectations for wild type sequence
    free_params = initialize_params(inits=inits, number_of_muts = len(mutID_dict))
    
    #write array of measured scores for calculating loss (not used by minimizer)
    measures = anp.array([scores[s] + offset for s in scores])

    #create output file and remove any existing files with same name
    with open(output_file,'w+') as file_out:
        #file_out.write("pos\taa\tKx_pred\tKc_pred\talpha_pred\tE_0f\tE_1f\tE_0b\tE_1b\tdG_pred\tdG_meas\titer\n")
        file_out.write('\t'.join([
                "pos", "aa", '\t'.join(quants),
                '\t'.join(state_names), "dG_pred",
                "dG_meas", 'iter']) + '\n')

    #reload data from previous optimization if necessary
    #(sets current i to whatever the last one was during previous optimization)
    if reload_file != "NA":
        sys.stderr.write("loading in estimates from %s ...\n" % reload_file)
        iter_start, free_params = load_params(reload_file, free_params, mutID_dict)
        reload_file = "NA"    
    else:
        iter_start = 0
    
    free_params = anp.reshape(free_params, -1)
    wt_energies = free_params[:num_states-1]
    old_grad = None
        
    ##########MAIN LOOOP##########
    for i in range(iter_start, int(max_iter)+1):
        #print(free_params)
        ###First, calculate metadate about current progress and write results###
        if i == iter_start:
            corr = "NA"
            step_size = "NA"
            iter_timer = "NA"
            fix_wt_flag = False
            norm_vector = find_normalization_vector(free_params, scores, mutID_dict, fix_wt = fix_wt_flag)

        else:
            #calculate correlation between predictions and measurements
            corr = round(float(pearsonr(predict_scores(free_params, scores, mutID_dict, num_threads, wt_energies), measures)[0])**2,3)
            end_clock = time.time()
            iter_timer = round(end_clock - start_clock,2)

        #calculate loss
        loss_new = mt_loss_func(free_params, measures, scores, mutID_dict, num_threads, wt_energies, chunk_size = 0, fix_wt = fix_wt_flag)
        #print to screen
        sys.stdout.write("loss_iter_%s=\t%s\tr^2=\t%s\tstep_size=\t%s\ttime=%s\n" % (i, round(loss_new,4), corr, step_size, iter_timer))
        start_clock = time.time()
        
        #write output information for this iteration to file        
        if i % recording_interval == 0:
            write_ouput_iter(output_file, free_params, mutID_dict, singles, offset, i, num_threads, wt_energies)

        ###Next, run a round of descent###
        new_grad = mt_gradient(free_params, scores, mutID_dict, offset, num_threads, wt_energies, chunk_size, fix_wt = fix_wt_flag)

        #Normalize the gradient by dividing by the relative number of observations
        #contributing to it. Mostly useful for preventing the wild type params
        #from swinging out of control
        new_grad = anp.array([new_grad[i] * norm_vector[i] for i in range(len(new_grad))])

        #Pick optimal direction to update in
        d_i = polak_ribiere_step(
                gradient_k1 = new_grad,
                gradient_k = old_grad
                )
        old_grad = new_grad
        
        #Pick optimal step size
        step_size = fast_line_search(free_params, d_i, measures, scores, mutID_dict, num_threads, wt_energies, chunk_size, offset)
        
        #update parameters
        free_params += d_i * step_size
        wt_energies = free_params[:num_states-1]
        
#################################END MAIN######################################




if __name__ == "__main__":

    import argparse
    import autograd.numpy as anp
    from autograd import grad
    import random
    from scipy.stats.stats import pearsonr
    from multiprocessing import Pool
    import time
    
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
    parser.add_argument("-t",
                  "--tolerance",
                  action = 'store',
                  dest="tolerance",
                  default="1E-6",
                  help="Program will break if decrease in loss is less than tolerance for 5 iterations in a row"
                  )
    parser.add_argument("-m",
                  "--maxIter",
                  action = 'store',
                  dest="max_iter",
                  default="1000",
                  help="Program will break after this many iterations of gradient descent"
                  )
    parser.add_argument("-n",
                  "--num-threads",
                  action = 'store',
                  dest="num_threads",
                  type=int,
                  default="8",
                  help="Number of threads for parallelizing gradients"
                  )
    parser.add_argument("-r",
                  "--reload-file",
                  action = 'store',
                  dest="reload_file",
                  type=str,
                  default="NA",
                  help="Load in a file from previous optimization to continue optimizing"
                  )
    parser.add_argument("-c",
                  "--chunk-size",
                  action = 'store',
                  dest="chunk_size",
                  type=int,
                  default="0",
                  help="Size of data chunks to use for training. Zero will use all the data every itteration."
                  )
    parser.add_argument("-e",
                  "--recording-interval",
                  action = 'store',
                  dest="recording_interval",
                  type=int,
                  default="1",
                  help="How many iterations to run between writing the current energies to file"
                  )


    options = parser.parse_args()

    main(options.input_scores, options.output_file, options.offset,
         options.tolerance, options.max_iter, options.num_threads,
         options.reload_file, options.chunk_size, options.recording_interval)

