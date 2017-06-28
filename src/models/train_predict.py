from __future__ import print_function, division

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import silhouette_score
from keras.layers import LSTM, Input, Dense, GRU
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, Callback
from sklearn.model_selection import train_test_split
from mlxtend.file_io import find_files

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns


def main(do_train, do_predict, model_str, kw_dict):
    """Main function of the module. 
    This builds or loads the network, trains the network and finally outputs
    predictions.
    """
    if model_str is None:
        model = build_model(kw_dict)
        kw_dict['modelname'] = make_model_name(kw_dict)
    else:
        model = load_model(model_str)
        kw_dict['modelname'] = make_model_name(model_str)
    if not os.path.exists(kw_dict['modeldir']):
        os.makedirs(kw_dict['modeldir'])
    kw_dict['basepath'] = os.path.join(kw_dict['modeldir'], kw_dict['modelname'])

    if do_train:
        model, hist = train(model, kw_dict)
        plot_training_hist(hist, kw_dict)

    if do_predict:
        predictions, idxs = predict(model, kw_dict)
        write_predictions(predictions, idxs, kw_dict)
    
    write_script_parameters(kw_dict)

def build_model(kw_dict):
    """Build a neural network according to the specifications in `kw_dict`"""
    n_nodes = kw_dict['n_nodes']
    if kw_dict['recurrent_unit'].lower() == 'gru':
        re_unit = GRU
    elif kw_dict['recurrent_unit'].lower() == 'lstm':
        re_unit = LSTM
    else:
        raise NotImplementedError

    model = Sequential()
    model.add(re_unit(units=n_nodes, input_shape=(None, 1)))
    if kw_dict['extra_dense']:
        model.add(Dense(units=n_nodes//2))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def make_model_name(model_info):
    """Figure out an appropriate name for the model, 
    given either a path or a dict with parameters"""
    if isinstance(model_info, str):
        # Split somehow
        filename = os.path.basename(model_info)
        params = filename.split('__')[:-1]
        modelname = '__'.join(params)
    else:
        # Generate from dict
        params = ['recurrent_unit', 'n_nodes', 'extra_dense', 'upper_cutoff', 'lower_cutoff']
        params_w_value = ['{}_{}'.format(x, model_info[x]) for x in params]
        modelname = '__'.join(params_w_value)
    return modelname
    
def train(model, kw_dict):
    """Train the model. Load data, split into training, validation and holdout. 
    Run the fit function, select the best epoch based on validation performance.
    Eval that model on the holdout set. """
    n_epochs = kw_dict['n_epochs']
    traces, labels = load_training_traces(kw_dict)
    weights = get_class_weights(labels)
    X_train, X_test, y_train, y_test = train_test_split(traces, labels, test_size=0.3, stratify=labels)
    X_valid, X_holdout, y_valid, y_holdout = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

    training_gen = data_generator(X_train, y_train, n_classes=1)
    valid_gen = data_generator(X_valid, y_valid, n_classes=1)
    holdout_gen = data_generator(X_holdout, y_holdout, n_classes=1)
    callbacks = get_callbacks(kw_dict)
    hist = model.fit_generator(training_gen, steps_per_epoch=len(X_train),
                    validation_data=valid_gen, validation_steps=len(X_test), epochs=n_epochs,
                    class_weight=weights, callbacks=callbacks, verbose=kw_dict['verbose'])
    model = load_best_training_model(kw_dict)

    losses_holdout = model.evaluate_generator(holdout_gen, steps=len(X_holdout))
    df_holdout = pd.DataFrame()
    df_holdout['holdout_loss'] = [losses_holdout[0]]
    df_holdout['holdout_acc'] = [losses_holdout[1]]
    df_holdout.to_csv(kw_dict['basepath'] + '__holdout.csv', index=False)
    return model, hist

def load_training_traces(kw_dict):
    """Find the folder where the labelled data lives. Split into "Molecule" or
    "Tunneling" (no molecule)"""
    root_path = kw_dict['datadir']
    upper_cut = kw_dict['upper_cutoff']
    lower_cut = kw_dict['lower_cutoff']

    pos_df = pd.read_csv(kw_dict['pos_idx_file'])
    pos_idxs = pos_df.idx.values
    pos_files = get_filenames_from_index(pos_idxs, kw_dict)
    pos_traces = get_traces_from_filenames(pos_files)
    pos_cut_traces, _ = preprocess(pos_traces, upper_cutoff=upper_cut, lower_cutoff=lower_cut)

    neg_df = pd.read_csv(kw_dict['neg_idx_file'])
    neg_idxs = neg_df.idx.values
    neg_files = get_filenames_from_index(neg_idxs, kw_dict)
    neg_traces = get_traces_from_filenames(neg_files)
    neg_cut_traces, _ = preprocess(neg_traces, upper_cutoff=upper_cut, lower_cutoff=lower_cut)

    training_traces = pos_cut_traces + neg_cut_traces
    labels = np.array([1] * len(pos_cut_traces) + [0] * len(neg_cut_traces))
    return training_traces, labels

def get_filenames_from_index(idxs, kw_dict):
    datadir = kw_dict['datadir']
    data_basename = '17_03_31_BP_4K_'
    filenames = [os.path.join(datadir, '17_03_31_BP_4K_{}.dat'.format(ii)) for ii in idxs]
    return filenames

def get_traces_from_filenames(filenames):
    read_rows = 4100
    traces = np.zeros(shape=(len(filenames), read_rows))
    for ii, f in enumerate(filenames):
        df = pd.read_csv(f, skiprows=1, nrows=read_rows, names='G')
        traces[ii] = df.G.values
    return traces

def get_class_weights(labels):
    """Make the 2 classes balanced in training, by weighing them"""
    total_samples = len(labels)
    weights = {total_samples / np.sum(labels == ii) for ii in range(2)}
    return weights


def preprocess(traces, upper_cutoff, lower_cutoff):
    """Cut each trace according to 2 cutoffs. Take the log"""
    processed = []
    good_idxs = []
    for ii, trace in enumerate(traces):
        first_idx = np.where(trace < upper_cutoff)[0][0]
        last_idx = np.where(trace < lower_cutoff)[0][0]
        cut_trace = trace[first_idx:last_idx]
        # neg_idx = np.where(cut_trace < 0)[0]
        # cut_trace[neg_idx] = lower_cutoff
        processed_trace = np.log(cut_trace)
        if len(processed_trace) == 0:
            continue
        else:
            good_idxs.append(ii)
            processed.append(processed_trace)
    assert len(processed) == len(good_idxs)
    return processed, good_idxs


def data_generator(traces, labels, n_classes=2):
    """Generator returning (trace, label)"""
    while True:
        for t, l in zip(traces, labels):
            t = t.reshape(1, -1, 1)
            l = l.reshape(-1, n_classes)
            yield (t, l)


def get_callbacks(kw_dict):
    """Construct the callbacks used in training"""
    basepath = kw_dict['basepath']
    model_chkpnt_filename = basepath + \
        '__epoch_{epoch:02d}__val_loss_{val_loss:.3f}.hdf5'
    model_best_filename = basepath + '__best.hdf5'
    checkpointer = ModelCheckpoint(
        monitor='val_loss', filepath=model_chkpnt_filename, save_best_only=True, mode='min')
    save_best = ModelCheckpoint(
        monitor='val_loss', filepath=model_best_filename, save_best_only=True, mode='min')
    csvlogger = CSVLogger(filename=basepath +
                            '__training.log', append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=30)
    callbacks = [checkpointer, csvlogger, early_stop, save_best]
    return callbacks


def plot_training_hist(hist, kw_dict):
    """From the `History` object from training, plot loss and accuracy"""
    basepath = kw_dict['basepath']
    h = hist.history
    plt.plot(h['loss'], label='Training loss')
    plt.plot(h['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(basepath + '__loss.png')
    plt.clf()

    plt.plot(h['acc'], label='Training acc')
    plt.plot(h['val_acc'], label='Validation acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(basepath + '__acc.png')
    plt.clf()

def load_best_training_model(kw_dict):
    """Given a modelname, we can find the network with best validation performance"""
    basepath = kw_dict['basepath']
    modelpath = basepath + '__best.hdf5'
    model = load_model(modelpath)
    return model


def predict(model, kw_dict):
    """Predict on the unsorted traces"""
    traces, idxs = load_predict_traces(kw_dict)
    n_samples = len(traces)
    trace_gen = data_generator(traces, np.arange(n_samples).reshape(-1,1), n_classes=1)
    predictions = model.predict_generator(trace_gen, steps=n_samples, verbose=kw_dict['verbose'])
    return predictions, idxs

def load_predict_traces(kw_dict):
    root_path = kw_dict['datadir']
    upper_cut = kw_dict['upper_cutoff']
    lower_cut = kw_dict['lower_cutoff']

    predict_df = pd.read_csv(kw_dict['predict_idx_file'])
    predict_idxs = predict_df.idx.values
    predict_files = get_filenames_from_index(predict_idxs, kw_dict)
    predict_traces = get_traces_from_filenames(predict_files)
    predict_cut_traces, good_idxs = preprocess(predict_traces, upper_cutoff=upper_cut, lower_cutoff=lower_cut)
    cut_idxs = predict_idxs[good_idxs]

    assert len(predict_cut_traces) == len(cut_idxs)
    return predict_cut_traces, cut_idxs


def get_trace_idx_from_filename(filename):
    """From the path to the trace datafile, get the index
    I assume a filename like 12_34_BP_4K_99.dat, where 99 is the trace index. """
    basename = os.path.basename(filename)
    idx_and_ext = basename.split('_')[-1]
    idx = idx_and_ext.split('.')[0]
    return int(idx)


def write_predictions(predictions, idxs, kw_dict):
    """Write predictions to a .csv for later analysis"""
    df = pd.DataFrame()
    df['trace_idx'] = np.array(idxs, dtype=int)
    df['pred'] = predictions

    path = kw_dict['basepath'] + '__predictions.csv' 
    df.to_csv(path, index=False)

def write_script_parameters(kw_dict):
    """Write the script parameters to a pickle file so they can be read later"""
    path = kw_dict['basepath'] + '__params.pickle'
    with open(path, 'wb') as f:
        pickle.dump(kw_dict, f)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_nodes', default=6, type=int)
    parser.add_argument('--n_epochs', default=4, type=int)
    parser.add_argument('--extra_dense', action='store_true', default=False)
    parser.add_argument('--recurrent_unit', default='lstm', type=str)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_predict', action='store_true')
    parser.add_argument('--load_model', type=str, default=None)
    parser.add_argument('--upper_cutoff', type=float, default=1e-1)
    parser.add_argument('--lower_cutoff', type=float, default=1e-6)
    parser.add_argument('--datadir', type=str, default='data/processed/all_traces')
    parser.add_argument('--modeldir', type=str, default='models/tun-mol')
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--jobid', type=str, default=None, help="Write slurm job-id here. Only to help find the output")
    parser.add_argument('--pos_idx_file', type=str, default='data/processed/molecular_index.csv')
    parser.add_argument('--neg_idx_file', type=str, default='data/processed/tunnel_index.csv')
    parser.add_argument('--predict_idx_file', type=str, default='data/processed/all_index.csv')

    args = parser.parse_args()
    kw_dict = vars(args)
    if args.seed is not None:
        np.random.seed(args.seed)
    main(args.do_train, args.do_predict, model_str=args.load_model, kw_dict=kw_dict)