import os
from os.path import join
# import argparse
from configargparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from astartes import train_test_split
from pathlib import Path
from copy import deepcopy

from prosmith.utils.kinase_utils import get_pipeline_seq, read_dataset, generate_input_df
from prosmith.preprocessing.smiles_embeddings import *
from prosmith.preprocessing.protein_embeddings import *
from prosmith.utils.gen_utils import save_config

def get_preprocessing_parser(**kwargs):
    parser = ArgumentParser(**kwargs)
    # dataset arguments
    parser.add_argument(
        "--dataset",
        default='0.5uM',
        choices=['0.5uM', '1.0uM', 'all'], # the all option combines the dosages and adds a dose embedding parameter in model
        help=f"Dataset: 0.5uM, 1.0uM (default: 0.5uM)"
    )
    parser.add_argument(
        "--task",
        default='regression',
        choices=['binary', 'multiclass', 'regression'],
        help=f"Task type: regression, binary, multiclass (default: regression)"
    )
    # for SMILES and sequence specifications
    parser.add_argument(
        "--smiles",
        default='canonical',
        choices=['canonical', 'isomeric_freebase'],
        help=f"SMILES: canonical, isomeric_freebase (default: canonical)"
    )
    parser.add_argument(
        "--no_isoform",
        action='store_true',
        help=f"By default isoforms are included with their full sequence. Can turn off to drop them from data."
        # only drops the non canonical isoform
    )
    parser.add_argument(
        "--full_sequence",
        action='store_true',
        help=f"By default only AAs near ATP binding domain are included, but can use full sequence up to 1024 length max."
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=10,
        help="The number of amino acids to include to the left and to the right of the ATP binding domain."
    )
    # ideally split by drugs to mimic new drug data
    parser.add_argument(
        "--split_type",
        default='scaffold',
        choices=['random', 'scaffold'],
        help="Splitting type. (default: scaffold)."
    )
    parser.add_argument(
        "--split_sizes",
        type=float,
        nargs=3,
        default=[0.8, 0.1, 0.1],
        help="Split proportions for train, validation, test sets (default: 0.8, 0.1, 0.1).",
    )
    parser.add_argument(
        "--train_val_split_seed",
        type=int,
        default=100,
        help="Random seed to use when splitting data into train/val/test sets. (default: 100)",
    )
    parser.add_argument(
        "--test_split_seed",
        type=int,
        default=2300,
        help="Random seed to use when splitting data into train/val/test sets. (default: 2300)",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Path where preprocessed dataset and embeddings will be saved.",
    )
    parser.add_argument(
        "--esm_version",
        default='2',
        choices=['1b', '2'],
        help=f"ESM model version: 1b, 2 (default: 2)"
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=1280,
        choices=[1280, 2560, 5120],
        help=f"ESM model embedding dim: 1280, 2560, 5120 (default: 1280)."
    )
    # parser.add_argument(
    #     "--prot_emb_no",
    #     default=10000, # can increase as we have lots of RAM - since we have <2000 proteins and smiles we should be ok
    #     type=int,
    #     help="Number of protein sequences in one dictionary.", # this determines subsets read in by dataloader
    # )
    # parser.add_argument(
    #     "--smiles_emb_no",
    #     default=200000,
    #     type=int,
    #     help="Number of SMILES strings in one dictionary.",
    # )
    return parser

if __name__ == "__main__":
    parser = get_preprocessing_parser()
    args = parser.parse_args()
    
    # first get our dataframe ready for model input
    df, kinases = read_dataset(args.dataset, no_isoform=args.no_isoform, smiles=args.smiles)
    input_df = generate_input_df(df, task=args.task)
    
    # get sequence strings - need dictionary that maps each name to sequence
    seq_dict = {}
    for kinase in tqdm(kinases):
        kinase_seq = get_pipeline_seq(kinase, 
                                      full_seq=args.full_sequence, 
                                      no_isoform=args.no_isoform, 
                                      start=22, 
                                      end=1919, 
                                      context_size=args.context_size, 
                                      max_length=1000)
        if kinase_seq is not None:
            seq_dict[kinase] = kinase_seq
        else:
            raise Exception(f'Something went wrong while calculating the sequence for {kinase}')

    input_df['Protein sequence'] = input_df['kinase'].apply(lambda kinase: seq_dict[kinase])
    
    # Get all unique protein Sequences and SMILES strings - more efficient way by just iterating over list
    all_sequences = list(seq_dict.values())
    all_smiles = list(set(df["SMILES"]))
    print(f"No. of different sequences: {len(all_sequences)}, No. of different SMILES strings: {len(all_smiles)}")

    # Data splitting by drug
    # first split to get test set with test seed
    train_val_smiles, test_smiles, _, _ = train_test_split(np.array(all_smiles), 
                                                           train_size=1 - args.split_sizes[2],
                                                           test_size=args.split_sizes[2],
                                                           sampler=args.split_type, 
                                                           random_state=args.test_split_seed)

    # then split train / val set from remaining
    val_size = args.split_sizes[1]/(args.split_sizes[0] + args.split_sizes[1])
    train_smiles, val_smiles, _, _ = train_test_split(train_val_smiles, 
                                                      train_size=1 - val_size,
                                                      test_size=val_size,
                                                      sampler=args.split_type, 
                                                      random_state=args.train_val_split_seed)

    # splitting the data (partition by drugs) - save train / val / test data to separate csv files
    train_df = input_df.loc[input_df["SMILES"].isin(train_smiles)]
    val_df = input_df.loc[input_df["SMILES"].isin(val_smiles)]
    test_df = input_df.loc[input_df["SMILES"].isin(test_smiles)]

    try:
        embed_path = args.save_path / 'embeddings'
        embed_path.mkdir(exist_ok=True, parents=True)
    except:
    	raise Exception('Unable to make save directory.')

    # save data to files
    train_df.to_csv(args.save_path / 'train.csv', index=False)
    val_df.to_csv(args.save_path / 'val.csv', index=False)
    test_df.to_csv(args.save_path / 'test.csv', index=False)
    
    # save config file also
    save_config(parser, args, args.save_path / 'config.toml')
    
    # embeddings contains all proteins and smiles across train / val / test split 
    print("Calculating protein embeddings:")
    # calculate_protein_embeddings(all_sequences, str(embed_path), args.prot_emb_no, esm_version=args.esm_version, embedding_size=args.embedding_size)
    calculate_protein_embeddings_fast(all_sequences, str(embed_path), esm_version=args.esm_version, embedding_size=args.embedding_size, toks_per_batch=4096, use_gpu=True)
    print("Calculating SMILES embeddings:")
    # calculate_smiles_embeddings(all_smiles, str(embed_path), args.smiles_emb_no)
    calculate_smiles_embeddings_fast(all_smiles, str(embed_path))
