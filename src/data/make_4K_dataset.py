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
parser.add_argument('--input_basename', type=str, default='17_03_31_BP_4K_')


def main(input_filepath, output_filepath, input_basename):
    """ Runs data processing scripts to turn raw data from (../external) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    make_index_files(input_filepath, output_filepath, input_basename)
    copy_all_data(input_filepath, output_filepath)


def make_index_files(input_filepath, output_filepath, input_basename):
    """Get trace indices from the raw trace files in a subfolder. Save the
    index files in ../processed"""
    tunnel_idxs = utils.get_index_in_folder(input_filepath, input_basename, 'Tunnel')
    bipy_stays_idxs = utils.get_index_in_folder(input_filepath, input_basename, 'BP_Stays')
    bipy_flips_idxs = utils.get_index_in_folder(input_filepath, input_basename, 'BP_FlipsOut')
    all_idxs = utils.get_index_in_folder(input_filepath, input_basename, 'UnSorted')

    molecular_idxs = bipy_stays_idxs + bipy_flips_idxs

    # Read index from csv
    path = os.path.join(input_filepath, 'BP_lowT_step_G.csv')
    df_G_step = pd.read_csv(path)
    bipy_with_step_idxs = df_G_step['BP with step'].values
    bipy_without_step_idxs = df_G_step['BP without step'].values

    # Write the index csv files
    utils.write_index_file(output_filepath, 'tunnel', tunnel_idxs)
    utils.write_index_file(output_filepath, 'bipy_stays', bipy_stays_idxs)
    utils.write_index_file(output_filepath, 'bipy_flips', bipy_flips_idxs)
    utils.write_index_file(output_filepath, 'all', all_idxs)
    utils.write_index_file(output_filepath, 'molecular', molecular_idxs)
    utils.write_index_file(output_filepath, 'bipy_with_step', bipy_with_step_idxs)
    utils.write_index_file(output_filepath, 'bipy_without_step', bipy_without_step_idxs)


def copy_all_data(input_filepath, output_filepath):
    """Copy data from from external folder to processed folder"""
    dirs = ['Tunnel', 'BP_FlipsOut', 'BP_Stays', 'UnSorted']
    outdir = os.path.join(output_filepath, '4K_traces')
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
