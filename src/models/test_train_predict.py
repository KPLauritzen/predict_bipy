import pytest
import src.models.train_predict as tp

def test_get_trace_idx_from_filename():
    # Simple test
    assert tp.get_trace_idx_from_filename('12_34_BP_4K_99.dat') == 99

    # Handle folders
    assert tp.get_trace_idx_from_filename('data/abc_123.4_4K_12.dat') == 12

def test_make_model_name():
    assert tp.make_model_name('good__network__best.hdf5') == 'good__network'

    assert tp.make_model_name('/long/path/before/good__network__best.hdf5') == 'good__network'

def test_build_model():
    kw_dict = {'n_nodes':4, 'recurrent_unit':'LSTM', 'extra_dense':True}

    model = tp.build_model(kw_dict)
    assert isinstance(model, tp.Sequential)

    kw_dict['recurrent_unit'] = 'awesome_rnn'
    with pytest.raises(NotImplementedError):
        tp.build_model(kw_dict)