"""This starts many SLURM jobs on steno, training a neural network with some set of parameters"""

import os
import itertools
import numpy as np

submit_path = os.path.expanduser('~/pulling/scripts/submit.py')
script_path = os.path.abspath('src/models/train_predict.py')



n_nodes = [4, 6, 8, 10]
dense = ['--extra_dense', '']
recurrent_units = ['lstm', 'gru']
upper_lower_cutoffs = [(1e-1, 1e-7), (1e-1, 1e-6)]
seeds = [1, 2, 3]


iterables = (n_nodes, dense, recurrent_units, upper_lower_cutoffs, seeds)
total_length = np.prod([len(x) for x in iterables])
print "You are about to submit {} jobs.".format(total_length)
var = raw_input("Is this OK? [y/n]: ")
if var == 'y':
    print "go ahead"
else:
    print "stopping"
    exit()


for  nodes, extra_dense, recur_unit, upper_lower, seed in itertools.product(*iterables):
    upper_cut, lower_cut = upper_lower
    jobname = 'bipy_tunmol__{}__{}nodes'.format(recur_unit, nodes)
    submit_string = """python {submit_path} --scriptname={script_path} --partition kemi_gemma3 --no_scratch --mail fail --jobname {jobname} --jobid --py_args='--do_train --do_predict --n_nodes {nodes} --n_epochs 300 {extra_dense} --recurrent_unit {recur_unit} --upper_cutoff {upper_cut} --lower_cutoff {lower_cut} --datadir data/processed/all_traces --modeldir models/tun-mol-seed-{seed} --pos_idx_file data/processed/molecular_index.csv --neg_idx_file data/processed/tunnel_index.csv --predict_idx_file data/processed/all_index.csv --seed {seed}'""".format(**locals())
    os.system(submit_string)
    os.system("sbatch submit.sl")
