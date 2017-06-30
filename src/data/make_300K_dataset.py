# -*- coding: utf-8 -*-
import os
import argparse
import logging
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from tqdm import tqdm
import make_dataset_utils as utils


parser = argparse.ArgumentParser()
parser.add_argument('--input_filepath', type=str, default='data/external')
parser.add_argument('--output_filepath', type=str, default='data/processed')
parser.add_argument('--input_basename', type=str, default='16_08_11_BP_')


def main(input_filepath, output_filepath, input_basename):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    make_index_files(input_filepath, output_filepath, input_basename)
    copy_all_data(input_filepath, output_filepath)


def make_index_files(input_filepath, output_filepath, input_basename):
    all_idxs = utils.get_index_in_folder(input_filepath, input_basename, 'BP_300K/')

    # Read index from csv
    path = os.path.join(input_filepath, 'BP_300K_tunmol.csv')
    df = pd.read_csv(path)
    tunnel_idxs = df['Tunnel'].values
    bp_idxs = df['BP present in Pull'].values

    # Andras full classification
    path = os.path.join(input_filepath, 'BP_300K_tunmol_full.csv')
    df = pd.read_csv(path)
    full_tunnel_idxs = df['Tunnel'].values
    full_bp_idxs = df['BP present in Pull'].values

    # Write the index csv files
    utils.write_index_file(output_filepath, '300K_all', all_idxs)
    utils.write_index_file(output_filepath, '300K_tunnel', tunnel_idxs)
    utils.write_index_file(output_filepath, '300K_molecular', bp_idxs)

    utils.write_index_file(output_filepath, '300K_tunnel_full', full_tunnel_idxs)
    utils.write_index_file(output_filepath, '300K_molecular_full', full_bp_idxs)


def copy_all_data(input_filepath, output_filepath):
    dirs = ['BP_300K']
    outdir = os.path.join(output_filepath, '300K_traces')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for d in dirs:
        full_dir = os.path.join(input_filepath, d)
        files = os.listdir(full_dir)
        for f in tqdm(files):
            full_filename = os.path.join(full_dir, f)
            utils.write_opening_closing(full_filename, outdir)

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    args = parser.parse_args()
    main(input_filepath=args.input_filepath, output_filepath=args.output_filepath, input_basename=args.input_basename)
