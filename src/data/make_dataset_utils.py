import os
import pandas as pd
import pickle
from mlxtend.file_io import find_files
def get_index_in_folder(input_filepath, input_basename, folder):
    path = os.path.join(input_filepath, folder)
    files = find_files(input_basename, path=path, check_ext='.dat')
    idxs = []
    for f in files:
        idxs.append(get_index_from_filename(f))
    return idxs


def get_index_from_filename(full_path):
    filename = os.path.basename(full_path)
    idx_and_ext = filename.split('_')[-1]
    idx = idx_and_ext.split('.')[0]
    return int(idx)

def write_index_file(output_filepath, index_name, idxs):
    out_path = os.path.join(output_filepath, index_name + '_index.csv')
    df = pd.DataFrame()
    df['idx'] = idxs
    df.to_csv(out_path, index=False)

def write_opening_closing(filename, outdir):
    with open(filename, 'r') as f:
        seen_params = 0
        open_trace = []
        close_trace = []
        params = {}
        for line in f:
            if line[0] == '#':
                continue
            if line[0] == 'X':
                seen_params += 1
                X0, dX = get_trace_params(line)
                if seen_params == 1:
                    params['open_X0'] = X0
                    params['open_dX'] = dX
                elif seen_params == 2:
                    params['close_X0'] = X0
                    params['close_dX'] = dX
                else:
                    raise ValueError
                continue
            else:
                value = float(line)
                if seen_params == 1:
                    open_trace.append(value)
                elif seen_params == 2:
                    close_trace.append(value)
                else:
                    raise ValueError
    basename = os.path.basename(filename)
    
    # Write open trace
    open_filename = os.path.join(outdir, 'open_' + basename)
    df = pd.DataFrame({'G':open_trace})
    df.to_csv(open_filename, index=False)

    # Write close trace
    close_filename = os.path.join(outdir, 'close_' + basename)
    df = pd.DataFrame({'G':close_trace})
    df.to_csv(close_filename, index=False)

    base, ext = os.path.splitext(basename)
    path = os.path.join(outdir, base)
    with open(path + '_params.pickle', 'w') as f:
        pickle.dump(params, f)
    return open_trace, close_trace

def get_trace_params(line):
    X0_part, dX_part = line.split(',')
    X0 = X0_part.split(' ')[-2]
    dX = dX_part.split(' ')[-2]
    return float(X0), float(dX)