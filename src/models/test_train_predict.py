import src.models.train_predict as tp

def test_get_trace_idx_from_filename():
    # Simple test
    assert tp.get_trace_idx_from_filename('12_34_BP_4K_99.dat') == 99

    # Handle folders
    assert tp.get_trace_idx_from_filename('data/abc_123.4_4K_12.dat') == 12
