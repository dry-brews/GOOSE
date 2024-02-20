#!/bin/bash
#SBATCH --job-name=pdz3_fit
#SBATCH --output=pdz3_fit.out
#SBATCH --error=pdz3_fit.err
#SBATCH --time=12:00:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --mem-per-cpu=2000

module load python/cpython-3.8.5

set -e
set -o pipefail
###Parameters###
base_directory=~/Workbench/Ranganathan_lab/Epistasis/GOOSE
scores_file=${base_directory}/data/deltaG_pdz3_table.tsv
output_dir=${base_directory}/output
model_file=${base_directory}/scripts/model3.py
split_script=${base_directory}/scripts/split_testing_training_051521.py
fit_script=${base_directory}/scripts/fit_energies_092221.py
doubleup_script=${base_directory}/scripts/write_doubles_091221.py
temp_dir=${base_directory}/temp_mod3/

#Step 1: move the things you need into the temp directory (including yourself)
if [[ -d ${temp_dir} ]]
then
  rm -r ${temp_dir}
fi

mkdir ${temp_dir}
cp ${fit_script} ${temp_dir}/fit_energies.py
cp ${split_script} ${temp_dir}/split_sets.py
cp ${model_file} ${temp_dir}/protein_model.py
cp ${doubleup_script} ${temp_dir}/make_doubles.py
cd ${temp_dir}

#Step 2: pre-process scores
echo splitting scores into testing and training sets...
python split_sets.py \
${scores_file}
echo done

#Step 3: fit energies to training dataset
echo fitting energies...
python fit_energies.py \
-i training_set.tsv \
-o ${output_dir}/GOOSE_model3_output.tsv \
-t 1E-6 \
-m 5000 \
-n 8 \
-w -8.23
echo done

#Step 4: calculate double scores for testing + training
python make_doubles.py \
${output_dir}/GOOSE_model3_output.tsv \
training_set.tsv \
> ${output_dir}/training_doubles_model_3.tsv

python make_doubles.py \
${output_dir}/GOOSE_model3_output.tsv \
testing_set.tsv \
> ${output_dir}/testing_doubles_model_3.tsv

rm -r ${temp_dir}
