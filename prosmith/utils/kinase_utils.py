import os
from os.path import join
import requests
from Bio import SeqIO
import pickle
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

ISOFORMS = ['LYN_B', 'PKCB2', 'PKG1B']

isoform_uniprot_id = {
    'LYN': 'P07948-1',
    'LYN_B': 'P07948-2',
    'PKCB1': 'P05771-1',
    'PKCB2': 'P05771-2', 
    'PKG1A': 'Q13976-1', 
    'PKG1B': 'Q13976-2'
}

selected_CDKs = ['CDK1_CYCLIN_B', 
                 'CDK2_CYCLIN_A', 
                 'CDK3_CYCLIN_E', 
                 'CDK4_CYCLIN_D1',
                 'CDK5_P35',
                 'CDK6_CYCLIN_D1',
                 'CDK7_CYCLIN_H',
                 'CDK8_CYCLIN_C',
                 'CDK9_CYCLIN_T1',
                 'CDK15_CYCLIN_A2',
                 'CDK19_CYCLIN_C']

atypical = ['STK22D_TSSK1', 
            'DNA_PK',
            'EEF2K',
            'MTOR_FRAP1',
            'PDK1_PDHK1',
            'PDK2_PDHK2',
            'PDK3_PDHK3', 
            'PDK4_PDHK4',
            'PKMZETA',
            'TRPM7_CHAK1']

# need to get path to file first
script_dir = os.path.dirname(os.path.abspath(__file__))
FASTA_id_path = join(script_dir, "data/FASTA_id_dict.pickle")
MSA_path = join(script_dir, "data/MSA.fasta")
ONE_DATASET_PATH = join(script_dir, 'data/kir_1.0uM_prosmith.csv')
HALF_DATASET_PATH = join(script_dir, 'data/kir_0.5uM_prosmith.csv')

with open(FASTA_id_path, 'rb') as file:
    FASTA_id = pickle.load(file)
    all_FASTA_ids = list(FASTA_id.values())

# modifications to mistakes in the FASTA for referencing full UniProt sequence:
modified_uniprot_id = {'AGC_SGK2/95-352': "Q9HBY8-1"}
modified_line_number = {'CMGC_CDK7/12-296': (12, 295), 'CMGC_CLK3/304-620': (156, 472),
                       'TKL_IRAK1/207-521': (207, 520)}

def create_empty_path(path):
	try:
		os.mkdir(path)
	except:
		pass

	all_files = os.listdir(path)
	for file in all_files:
		os.remove(join(path, file))

def get_full_seq(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        # Parse the FASTA format
        lines = fasta_data.split('\n')
        sequence = ''.join(lines[1:])  # Skip the first line which is the header
        return sequence
    else:
        raise Exception(f"Failed to fetch data for UniProt ID {uniprot_id}: HTTP Status code {response.status_code}")

def get_pipeline_seq(kinase, full_seq=False, no_isoform=False, start=22, end=1919, context_size=10, max_length=1000):
    """Given kinase name return the desired amino acid sequence string using the sequencing parameters."""
    def get_ATP_site(msa_seq, uniprot_id, lines, start, end, context_size, max_length):
        """Get the amino acid sequence around the ATP binding site.
        Parameters
        ----------
        msa_seq : str
            MSA amino acid sequence
        uniprot_id : str
            Uniprot identifier for downloading full sequence
        lines : (int, int)
            tuple of integers denoting the start and end of the MSA sequence in the full sequence on Uniprot
        start : int
            starting index of the ATP binding site
        end : int
            end (inclusive) index of the ATP binding site
        context_size : int
            number of additional amino acids to include at the ends of the ATP binding site
        max_length: int
            ensure sequence length does not exceed max length
    
        Returns
        -------
        str
        """
        # get middle portion first
        middle_seq = msa_seq[start:end+1].replace('-', '')
    
        # cut depending on context size
        if context_size > 0:
            left_seq = msa_seq[:start].replace('-', '')
            right_seq = msa_seq[end+1:].replace('-', '')
    
            # pull in full sequence if context size extends beyond the FASTA file limits
            if context_size > len(left_seq) or context_size > len(right_seq):
                full_seq = get_full_seq(uniprot_id)
                left_seq = full_seq[:lines[0]-1+len(left_seq)]
                right_seq = full_seq[lines[1]-len(right_seq):]

            # for infinite context (full sequence)
            if context_size == np.inf:
                if len(full_seq) <= max_length:
                    return full_seq
                else:
                    # get as many amino acids on both sizes as possible up to max length
                    seq_room = max_length - len(middle_seq)
                    if len(left_seq) < seq_room // 2:
                        left_context = left_seq
                        right_context = right_seq[:seq_room - len(left_seq)]
                    elif len(right_seq) < seq_room // 2:
                        left_context = left_seq[-(seq_room - len(right_seq)):]
                        right_context = right_seq
                    else:
                        left_context = left_seq[-(seq_room // 2):]
                        right_context = right_seq[:(seq_room // 2)]
            else:
                left_context = left_seq[-context_size:]
                right_context = right_seq[:context_size]
        else:
            left_context = ''
            right_context = ''
        return left_context + middle_seq + right_context

    # check that kinase has an MSA entry
    if FASTA_id[kinase] is None:
        raise Exception(f'Kinase {kinase} does not have MSA entry.')
        
    # first process separately if kinase is an isoform in which case full sequence is used (for both isoforms)
    if not no_isoform and kinase in isoform_uniprot_id:
        # note: here assuming isoform length should all be less than max
        return get_full_seq(isoform_uniprot_id[kinase]) 

    # search through the FASTA file for the entry
    p1 = re.compile('\S+ \S+ \S+ (.*)')
    p2 = re.compile('[A-Z0-9_]+/([0-9]+)-([0-9]+)')
    for record in SeqIO.parse(MSA_path, "fasta"):
        if record.id == '0ANNOTATION/1-2208':
            continue
            
        if record.id == FASTA_id[kinase]:
            # get uniprot id and line numbers for full sequence
            uniprot_id = p1.match(record.description).group(1) if \
                record.id not in modified_uniprot_id else modified_uniprot_id[record.id]
            lines = (int(p2.match(record.description).group(1)), int(p2.match(record.description).group(2))) if \
                record.id not in modified_line_number else modified_line_number[record.id]

            if full_seq:
                # for full seq still need to ensure within max length
                return get_ATP_site(str(record.seq).upper(), uniprot_id, lines, 22, 1919, np.inf, max_length)
            else:
                # Start 22, ALC END 1919
                return get_ATP_site(str(record.seq).upper(), uniprot_id, lines, 22, 1919, context_size, max_length)

    raise Exception(f'Sequence for {kinase} not found from FASTA file!')

def neutralize_charges(mol):
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(mol)

def convert_to_free_base_and_neutralize(smiles):
    remover = SaltRemover.SaltRemover()

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Remove salts
        mol_free_base = remover.StripMol(mol, dontRemoveEverything=True)
        # Neutralize charges
        mol_neutral = neutralize_charges(mol_free_base)
        # Convert molecule back to SMILES
        free_base_smiles = Chem.MolToSmiles(mol_neutral, isomericSmiles=True)
    else:
        raise Exception('Error while converting to mol from smiles.')
    return free_base_smiles

def read_dataset(dataset, no_isoform=False, smiles='canonical'):
    """Fetch kinase dataset dataframe and all kinases."""
    if dataset == '0.5uM':
        df = pd.read_csv(HALF_DATASET_PATH)
    elif dataset == '1.0uM':
        df = pd.read_csv(ONE_DATASET_PATH)
    elif dataset == 'all':
        raise NotImplementedError()
    else:
        raise Exception('Dataset must be one of these options: 0.5uM, 1.0uM, all.')

    if no_isoform:
        df = df.drop(columns=ISOFORMS)

    # also need to drop excess CDK-cyclin complexes
    drop_CDKs = []
    p = re.compile('CDK[0-9]+_.*')
    for col in df.columns:
        if p.match(col) and col not in selected_CDKs:
            drop_CDKs.append(col)
    df = df.drop(columns=drop_CDKs)

    # drop atypical kinases with no MSA
    df = df.drop(columns=atypical, errors='ignore')

    if smiles == 'canonical':
        df['SMILES'] = df['cSMILES']
        df = df.drop(columns=['cSMILES'])
    else:
        # isomeric freebase
        df['SMILES'] = df['SMILES'].apply(lambda x: convert_to_free_base_and_neutralize(str(x)))
        df = df.drop(columns=['cSMILES'])

    kinases = list(df.columns[2:])
    return df, kinases

def multiclass_threshold(x, high, med):
    if x <= high:
        return 2
    elif high < x <= med:
        return 1
    else:
        return 0

def generate_input_df(df, task='regression'):
    """Melt the kinase dataframe for model input.
    Parameters
    ----------
    df : pd.DataFrame
        Should have columns in format Compound, SMILES, kinase_1, kinase_2, ..., kinase_N

    Returns
    -------
    melted_df : pd.DataFrame
    """
    tmp = df.copy()
    
    # adding order column necessary to preserve correct shape with missing values and also get correct order
    tmp['order'] = range(len(tmp))
    melted_df = pd.melt(tmp, id_vars=['Compound', 'SMILES', 'order'], var_name='kinase', value_name='output')
    melted_df = melted_df.dropna(subset=['output'])

    if task == 'binary':
        melted_df['output'] = melted_df['output'].apply(lambda x: 1 if x <= 30 else 0)
    elif task == 'multiclass':
        melted_df['output'] = melted_df['output'].apply(lambda x: multiclass_threshold(x, 30, 60))
    
    return melted_df
    
def get_output_df(df):
    """Pivot the model input dataframe back into original form.
    Parameters
    ----------
    df : pd.DataFrame
        Should have columns in format Compound, SMILES, order, kinase, output
        
    Returns
    -------
    wide_df : pd.DataFrame
    """
    wide_df = df.pivot(index=['Compound', 'SMILES', 'order'], columns='kinase', values='output').reset_index()
    
    # Remove the 'order' column and sort by it to maintain the original row order
    wide_df = wide_df.sort_values(by='order').drop(columns=['order']).reset_index(drop=True)
    
    # Flatten the columns
    wide_df.columns.name = None  # remove the columns name
    wide_df.columns = wide_df.columns.to_list()  # flatten the columns if necessary
    return wide_df
