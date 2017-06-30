"""This starts many SLURM jobs on steno, training a neural network with some set of parameters"""

import os
import itertools
import numpy as np

submit_path = os.path.expanduser('~/pulling/scripts/submit.py')
script_path = os.path.abspath('src/models/train_predict.py')



n_nodes = [6, 10, 20]
dense = ['--extra_dense', '']
recurrent_units = ['lstm', 'gru']
upper_lower_cutoffs = [(1e-1, 1e-6), (2.0, 1e-6)]
seeds = [1, 2, 3]
#training_used = np.linspace(0.1, 1.0, num=4)
training_used = [1.0]
smoothings = [1, 2, 5]


iterables = (n_nodes, dense, recurrent_units, upper_lower_cutoffs, seeds, training_used, smoothings)
total_length = np.prod([len(x) for x in iterables])
print "You are about to submit {} jobs.".format(total_length)
var = raw_input("Is this OK? [y/n]: ")
if var == 'y':
    print "go ahead"
else:
    print "stopping"
    exit()


for  nodes, extra_dense, recur_unit, upper_lower, seed, train_data_used, smoothing in itertools.product(*iterables):
    upper_cut, lower_cut = upper_lower
    datafile_basename = 'open_16_08_11_BP_{}.dat'
    jobname = 'bipy_300K__{}__{}nodes'.format(recur_unit, nodes)
    submit_string = """python {submit_path} --scriptname={script_path} --partition kemi_gemma3 --no_scratch --mail fail --jobname {jobname} --jobid --py_args='--do_train --do_predict --n_nodes {nodes} --n_epochs 300 {extra_dense} --recurrent_unit {recur_unit} --upper_cutoff {upper_cut} --lower_cutoff {lower_cut} --datadir data/processed/300K_traces --modeldir models/300K-tun-mol-seed-{seed}-frac-train-{train_data_used} --datafile_basename {datafile_basename} --pos_idx_file data/processed/300K_molecular_index.csv --neg_idx_file data/processed/300K_tunnel_index.csv --predict_idx_file data/processed/300K_all_index.csv --seed {seed} --fraction_training_data_used {train_data_used} --smoothing {smoothing}'""".format(**locals())
    os.system(submit_string)
    os.system("sbatch submit.sl")
