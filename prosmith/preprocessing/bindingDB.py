import os
from os.path import join
# import argparse
from configargparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from prosmith.utils.kinase_utils import get_pipeline_seq, read_dataset, generate_input_df
from prosmith.preprocessing.smiles_embeddings import *
from prosmith.preprocessing.protein_embeddings import *
from prosmith.utils.gen_utils import save_config

# preprocess bindingDB for pretraining
def get_bindingdb_parser(**kwargs):
    parser = ArgumentParser(**kwargs)
    # for SMILES and sequence specifications
    parser.add_argument(
        "--save_path",
        type=Path,
        required=True,
        help="Path where preprocessed dataset and embeddings will be saved.",
    )
    parser.add_argument(
        "--use_gpu",
        action='store_true',
        help="Whether to use gpu for protein sequence embedding model.",
    )
    parser.add_argument(
        "--toks_per_batch",
        type=int,
        default=2048,
        required=True,
        help="Tokens per batch for batching sequences.",
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
    parser.add_argument(
        "--prot_emb_no",
        default=2000, # can increase as we have lots of RAM - since we have <2000 proteins and smiles we should be ok
        type=int,
        help="Number of protein sequences in one dictionary.", # this determines subsets read in by dataloader
    )
    parser.add_argument(
        "--smiles_emb_no",
        default=2000,
        type=int,
        help="Number of SMILES strings in one dictionary.",
    )
    return parser

if __name__ == "__main__":
    parser = get_bindingdb_parser()
    args = parser.parse_args()
    
    # read in and concat the dataframes
    test_df = pd.read_csv(str(args.save_path / 'cleaned_BindingDB_IC50_test.csv'))
    train_df = pd.read_csv(str(args.save_path / 'cleaned_BindingDB_IC50_train.csv'))
    df = pd.concat([train_df, test_df])
    
    # Get all unique protein Sequences and SMILES strings - more efficient way by just iterating over list
    all_sequences = list(set(df['Protein sequence']))
    all_smiles = list(set(df["SMILES"]))
    print(f"No. of different sequences: {len(all_sequences)}, No. of different SMILES strings: {len(all_smiles)}")

    try:
        embed_path = args.save_path / f'embeddings_{str(args.embedding_size)}_v{args.esm_version}'
        embed_path.mkdir(exist_ok=True, parents=True)
    except:
    	raise Exception('Unable to make save directory.')
    
    # embeddings contains all proteins and smiles across train / val / test split 
    print("Calculating protein embeddings:")
    calculate_protein_embeddings(all_sequences, str(embed_path), args.prot_emb_no, esm_version=args.esm_version, embedding_size=args.embedding_size, toks_per_batch=args.toks_per_batch, use_gpu=args.use_gpu)
    print("Calculating SMILES embeddings:")
    calculate_smiles_embeddings(all_smiles, str(embed_path), args.smiles_emb_no)
