"""
Parses outputted predictions, and writes indices of all positive traces,
i.e. P(trace == pos) > 0.5
"""
import pandas as pd

def main(predictions_file, index_output_file):
    pos_trace_idxs = get_pos_trace_idxs(predictions_file)
    write_trace_idxs(pos_trace_idxs, index_output_file)


def get_pos_trace_idxs(csv_file):
    df = pd.read_csv(csv_file)
    df_pos = df[df.pred > 0.5]
    pos_idx = df_pos.trace_idx.values
    return pos_idx


def write_trace_idxs(idxs, outfile):
    df = pd.DataFrame()
    df['idx'] = idxs
    df.to_csv(outfile, index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file', default='models/tun-mol/xx__predictions.csv')
    parser.add_argument('--index_output_file', default='data/processed/predicted_index.csv')

    args = parser.parse_args()

    main(args.predictions_file, args.index_output_file)
