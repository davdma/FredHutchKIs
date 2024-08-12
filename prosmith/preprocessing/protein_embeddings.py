import pathlib
import torch
import os
from os.path import join
import shutil

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained
from prosmith.utils.kinase_utils import create_empty_path
from Bio import SeqIO

def calculate_protein_embeddings(all_sequences, outpath, prot_emb_no = 1000, esm_version = '2', embedding_size=1280, toks_per_batch=4096, use_gpu=True):
    assert embedding_size in [1280, 2560, 5120]
    create_empty_path(join(outpath, "Protein"))
    fasta_file = join(outpath, "all_sequences.fasta")
    create_fasta_file(all_sequences, fasta_file)

    # Swapping to ESM2 - same embedding size of 1280
    if esm_version == '1b':
        if embedding_size == 1280:
            esm_model = "esm1b_t33_650M_UR50S"
        else:
            raise Exception('Embedding size outside of 1280 not supported for ESM 1b.')
    else:
        if embedding_size == 1280:
            esm_model = 'esm2_t33_650M_UR50D'
        elif embedding_size == 2560:
            esm_model = 'esm2_t36_3B_UR50D'
        else:
            esm_model = 'esm2_t48_15B_UR50D'
            
    print('Using ESM:', esm_model)
    model, alphabet = pretrained.load_model_and_alphabet(esm_model) # prev: "esm1b_t33_650M_UR50S"
    model.eval()
    if torch.cuda.is_available() and use_gpu:
        model = model.cuda()
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches)

    output_dir = join(outpath, "Protein", "temp")
    create_empty_path(output_dir)

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and use_gpu:
                toks = toks.to(device="cuda", non_blocking=True)

            # The model is trained on truncated sequences and passing longer ones in at
            # infernce will cause an error. See https://github.com/facebookresearch/esm/issues/21
            toks = toks[:, :1022]

            out = model(toks, repr_layers=[33], return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            

            for i, label in enumerate(labels):
                output_file = join(output_dir, label + ".pt")
                
                
                result = {"label": label}
                result["representations"] = {
                    layer: t[i, 1:len(strs[i]) + 1].clone()
                    for layer, t in representations.items()
                }
                
                torch.save(result, output_file)
    merge_protein_emb_files(output_dir, outpath, fasta_file, prot_emb_no)


def merge_protein_emb_files(output_dir, outpath, fasta_file, prot_emb_no):
    new_dict = {}

    version = 0
    fasta_sequences = SeqIO.parse(open(fasta_file),'fasta')

    for k, fasta in enumerate(fasta_sequences):

        if k % prot_emb_no == 0 and k > 0:
            torch.save(new_dict, join(outpath, "Protein", "Protein_embeddings_V"+str(version)+".pt"))
            new_dict = {}
            version +=1

        name, sequence = fasta.id, str(fasta.seq)
        rep_dict = torch.load(join(output_dir, name +".pt"))
        new_dict[sequence] = rep_dict["representations"][33].numpy()

    torch.save(new_dict, join(outpath, "Protein", "Protein_embeddings_V"+str(version)+".pt"))

    shutil.rmtree(output_dir)


def create_fasta_file(sequences, filename):
    ofile = open(filename, "w")
    # the index of sequence used as label here
    for k, seq in enumerate(sequences):
        ofile.write(">" + str(k) + "\n" + seq[:1018]  + "\n")
    ofile.close()





