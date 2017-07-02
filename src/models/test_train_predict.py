import pytest
import src.models.train_predict as tp
import pandas as pd
import numpy as np
def test_get_trace_idx_from_filename():
    # Simple test
    assert tp.get_trace_idx_from_filename('12_34_BP_4K_99.dat') == 99

    # Handle folders
    assert tp.get_trace_idx_from_filename('data/abc_123.4_4K_12.dat') == 12

def test_make_model_name():
    assert tp.make_model_name('good__network__best.hdf5') == 'good__network'

    assert tp.make_model_name('/long/path/before/good__network__best.hdf5') == 'good__network'

def test_build_model():
    kw_dict = {'n_nodes':4, 'recurrent_unit':'LSTM', 'extra_dense':True, 'dropout': 0.1, 'recurrent_dropout': 0.5}

    model = tp.build_model(kw_dict)
    assert isinstance(model, tp.Sequential)

    kw_dict['recurrent_unit'] = 'awesome_rnn'
    with pytest.raises(NotImplementedError):
        tp.build_model(kw_dict)
    return model

def test_train_predict(tmpdir):
    basename = 'testing{}.dat'
    pos_idx_file, neg_idx_file, datapath = make_artificial_data(tmpdir=tmpdir, basename=basename)
    model = test_build_model()
    kw_dict = {
        'n_epochs' : 1,
        'seed' : 1,
        'upper_cutoff' : 1.0,
        'lower_cutoff' : 0.1,
        'pos_idx_file' : pos_idx_file,
        'neg_idx_file' : neg_idx_file,
        'predict_idx_file' : pos_idx_file,
        'datadir' : datapath,
        'basepath' : datapath,
        'verbose': 0,
        'datafile_basename': basename,
        'fraction_training_data_used': 0.7,
        'smoothing': 10
    }
    model, _ = tp.train(model, kw_dict)
    tp.predict(model, kw_dict)
    tmpdir.remove()


def make_artificial_data(tmpdir, basename='17_03_31_BP_4K_{}.dat'):
    n_traces = 30
    all_idx = range(n_traces)
    pos_idx = all_idx[:n_traces/2]
    pos_idx_file = tmpdir.join('pos_idx')
    f = pos_idx_file.open('w')
    pd.DataFrame({'idx':pos_idx}).to_csv(f)

    neg_idx = all_idx[n_traces/2:]
    neg_idx_file = tmpdir.join('neg_idx')
    f = neg_idx_file.open('w')
    pd.DataFrame({'idx':neg_idx}).to_csv(f)
    for ii in all_idx:
        datapath = tmpdir.join(basename.format(ii)).open('w')
        pd.DataFrame({'G':np.random.random(100)}).to_csv(datapath)
    pos_path = '/' + pos_idx_file.relto('/')
    pos_path = pos_idx_file.realpath()
    neg_path = '/' + neg_idx_file.relto('/')
    dirpath = '/' + tmpdir.relto('/')
    return pos_path, neg_path, dirpath

