from __future__ import print_function
import pandas as pd
from mlxtend.file_io import find_files
import os
from tqdm import tqdm


def main(modeldir):
    contents = os.listdir(modeldir) # ls modeldir
    paths = [os.path.join(modeldir, el) for el in contents] # modeldir/el
    dirs = [el for el in paths if os.path.isdir(el)] # only if modeldir/el is a dir

    for subdir in dirs:
        dirname = os.path.basename(subdir)
        csvpath = os.path.join(modeldir, dirname + '-performance.csv')
        print("Looking at " + subdir)
        results_dict = get_results_from_training(subdir)
        if results_dict is None:
            print("No records found")
            continue
        print("writing to " + csvpath)
        write_performance(results_dict, csvpath)

def get_results_from_training(path):

    files = find_files('training', path=path, recursive=True, check_ext='.log')
    if len(files) == 0:
        return None
    header = ['network', 'n_nodes', 'extra_dense', 'upper_cutoff', 'lower_cutoff', 'smoothing', 'basename', 'best_val_loss', 'best_val_acc', 'best_epoch', 'holdout_acc', 'holdout_loss']
    missing_holdout = None
    result_dict = {el:list() for el in header}
    for logpath in tqdm(files):
        #do analysis
        try:
            df_log = pd.read_csv(logpath)
        except IOError:
            tqdm.write("Can't read: " + logpath)
            continue
        idx = find_best_epoch(df_log)
        best_val_loss, best_val_acc = df_log.loc[idx, ['val_loss', 'val_acc']]
        log_filename = os.path.basename(logpath)
        basename, _ = os.path.splitext(log_filename)
        params = basename.split('__')[:-1]
        basename = '__'.join(params)
        params.append(basename)
        params.append(best_val_loss)
        params.append(best_val_acc)
        params.append(idx)

        try:
            # Get holdout performance
            holdout_acc, holdout_loss = get_holdout_performance(logpath)
            params.append(holdout_acc)
            params.append(holdout_loss)
        except:
            if missing_holdout is None:
                tqdm.write("At least one trace missing holdout performance")
                missing_holdout = True
            continue

        for head, par in zip(header, params):
            result_dict[head].append(par)
    return result_dict

def get_holdout_performance(logpath):
    """
    Open the csv file with the holdout performance
    """
    len_ending = len('__training.log')
    basename = logpath[:-len_ending]
    performance_path = basename + '__holdout.csv'
    df = pd.read_csv(performance_path)
    acc = df.loc[0, 'holdout_acc']
    loss = df.loc[0, 'holdout_loss']
    return acc, loss


def find_best_epoch(df):
    """Get the epoch index with the lowest validation loss"""
    idx = df.val_loss.idxmin()
    return idx

def write_performance(results_dict, out_path):
    df_results = pd.DataFrame(results_dict)
    df_results.to_csv(out_path, index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', default='models/')
    args = parser.parse_args()

    main(args.modeldir)