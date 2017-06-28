from __future__ import print_function
import pandas as pd
from mlxtend.file_io import find_files
import os
from tqdm import tqdm


def main(modeldir, out_path):
    results_dict = get_results_from_training(modeldir)
    write_performance(results_dict, out_path)

def get_results_from_training(path):
    files = find_files('training', path=path, recursive=False, check_ext='.log')
    assert len(files) > 0
    header = ['network', 'n_nodes', 'extra_dense', 'upper_cutoff', 'lower_cutoff', 'basename', 'best_val_loss', 'best_val_acc', 'best_epoch', 'holdout_acc', 'holdout_loss']

    result_dict = {el:list() for el in header}
    for logpath in tqdm(files):
        #do analysis
        try:
            df_log = pd.read_csv(logpath)
        except IOError:
            print("Can't read: " + logpath)
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
            print("Can't read evaluation performance")
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
    parser.add_argument('--modeldir')
    parser.add_argument('--out_path')
    args = parser.parse_args()

    main(args.modeldir, args.out_path)