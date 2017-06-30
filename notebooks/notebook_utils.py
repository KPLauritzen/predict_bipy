import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_average_df(seeds, csv_basename, model_root, extra_params=None):
    if extra_params is None:
        extra_params = ['extra_dense', 'lower_cutoff', 'upper_cutoff', 'n_nodes', 'network' ]
    # Load csvs
    dfs = []
    for seed in seeds:
        csv_path = model_root + csv_basename.format(seed)
        df = pd.read_csv(csv_path)
        df = df.sort_values(by='basename').reset_index(drop=True)
        dfs.append(df)
    df_orig = dfs[0].copy()

    # Make a clean df to merge in
    df_combined = df_orig[['basename']].copy()
    for df in dfs:
        assert len(df) == len(df_combined)
        df_combined = pd.merge(df_combined, df, on='basename')

    # Make a clean df to get the average
    df_ave = df_orig[['basename'] + extra_params].copy()
    desired_cols = ['best_val_acc', 'best_val_loss', 'holdout_acc', 'holdout_loss']
    for col in desired_cols:
        df_filter = df_combined.filter(regex=col)
        ave = df_filter.mean(axis=1)
        df_ave[col] = ave

    df_ave = sort_df(df_ave)
    return df_ave

def sort_df(df, by='holdout_loss', rank=True):
    df.sort_values(by=by, inplace=True)
    if rank:
        df['rank'] = range(len(df))
    return df

def plot_ranked_performance(df, top=None):
    df = sort_df(df)
    if top is not None:
        df = df[df['rank'] < top]
    df.plot(x='rank',y=['holdout_acc', 'holdout_loss', 'best_val_acc', 'best_val_loss'])

def plot_trace(idx, datafile_basename, datadir):
    upper_cutoff = 2
    lower_cutoff = 1e-5
    trace = pd.read_csv(datadir + datafile_basename.format(idx)).G.values 
    cut_trace = np.where((trace < upper_cutoff) & (trace > lower_cutoff))[0] 
    trace = trace[cut_trace]
    plt.plot(trace)
    plt.xlabel('# datapoints')
    plt.ylabel('Conductance (G0')
    plt.title(datafile_basename.format(idx))
    plt.yscale('log')
    #plt.xlim([0, 400])
    plt.ylim([lower_cutoff, upper_cutoff])