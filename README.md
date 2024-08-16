# FredHutchKIs
![image](https://github.com/user-attachments/assets/8f0c4a10-49ca-4518-8ad1-b1c31d7e0793)

### Description: 
Using ProSmith<sup>10</sup> multimodal transformer networks for modeling kinase inhibitor-kinase small molecule to protein interaction and inhibition.

### Abstract:
Targeting key drivers of cancer proliferation and progression in protein kinases through small-molecule inhibitors have shown significant promise in cancer therapy. Beyond current FDA approved drugs, the next generation of kinase inhibitors may offer more potent or selective inhibition that current drugs do not have. To accelerate such drug discovery, we need strong and reliable predictive models for kinase inhibitor polypharmacology that can replace experimental screens with fast in silico screening. Given that kinase inhibition mainly occurs through binding in or near the ATP binding pocket<sup>13</sup>, we hypothesized that previous modeling efforts that used only kinase inhibitor chemical structures could be improved by incorporating kinase ATP binding site amino acid sequence information. We used a separate dataset of 383 kinase inhibitors in addition to our FDA dataset and obtained the amino acid sequence for each of the protein kinases. We then obtained protein sequence embeddings from the ESM<sup>2</sup> models built by Meta and trained the transformer network model ProSmith on the dataset. The preliminary results show that the ProSmith model which utilizes both drug and kinase information can potentially outperform previous state-of-the-art chemical property prediction model Chemprop<sup>11</sup>, and points towards the usefulness of protein language models in the problem of kinase inhibitor and kinase interactions, though more work still needs to be done to evaluate the its true performance.

![new_dataset2_fixed](https://github.com/user-attachments/assets/bff06d25-89e0-4a09-9e70-cc347598ff85)
**Figure 1: Dendrogram heatmap of new FDA drug dataset shows the broad range of kinase inhibition profiles.** The new dataset represents 92 FDA approved (or currently undergoing approval) drugs and their inhibitions across 393 wild type human kinases at 1.0uM dose. This is in addition to a larger proprietary dataset of 383 kinase inhibitors and 298 kinases at 0.5uM dose.

### Implementation
For preprocessing, we used the multiple sequence alignment done by Modi and Dunbrack<sup>1</sup> to obtain the amino acid sequence of the ATP binding site regions of each kinase containing the DFG motif and active residues that formed the catalytic region. We allowed for a `context_size` hyperparameter that determined how many of the surrounding amino acids of the catalytic region to include, up to a full sequence (1000 length maximum). We then extracted the token embeddings of each sequence from the ESM model (we tested v1b and v2 as well as different embedding sizes), and used it as the input alongside the SMILES embeddings from ChemBERTa. The model architecture is taken directly from the ProSmith codebase, but we made modifications to the dataloading and training loops to improve efficiency.

![image](https://github.com/user-attachments/assets/0811e5e3-aac5-436b-a1bf-3913c0d48318)
**Figure 2: Project modeling workflow.** We obtained the amino acid sequence of the catalytic region of each kinase through MSA. Prior to running the model, we validated the use of the ATP binding site sequence by making structural predictions using the sequences in ESMFold, and aligning them with the crystal structure of the kinase - the quality of the alignments showed that modeling the ATP binding site without the N and C terminal regions of the kinase is still viable.

### Works Cited:
1. Modi, V. and Dunbrack, R.L. (2019). Structurally-Validated Multiple Sequence Alignment of 497 Human Protein Kinase Domains. Sci Rep 2019, 9: 19790.
2. Rives, A., et al. (2019). Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. PNAS, 118(15):  e2016239118.
3. Mahajan, N.P., et al. (2018). Blockade of ACK1/TNK2 To Squelch the Survival of Prostate Cancer Stem-like Cells. Sci Rep, 8: 1954.
4. Kurioka, D., et al. (2014). NEK9-dependent proliferation of cancer cells lacking functional p53. Sci Rep, 4: 6111.
5. Karnan, S., et al. (2023). CAMK2D: a novel molecular target for BAP1-deficient malignant mesothelioma. Cell Death Discovery, 9: 257.
6. Quadri, R., et al. (2022). Roles and regulation of Haspin kinase and its impact on carcinogenesis. Cellular Signaling, 93: 110303.
7. Bennett, J. and Starczynowski, D.T. (2022). IRAK1 and IRAK4 as emerging therapeutic targets in hematologic malignancies. Curr Opin Hematol, 29(1): 8-19.
8. Torres-Ayuso, P., et al. (2021). TNIK Is a Therapeutic Target in Lung Squamous Cell Carcinoma and Regulates FAK Activation through Merlin. Cancer Discovery, 11(6): 1411-1423.
9. Eid S., et al. (2017). KinMap: a web-based tool for interactive navigation through human kinome data. BMC Bioinformatics, 18:16.
10. Kroll, A., et al. (2024). A multimodal Transformer Network for protein-small molecule interactions enhances predictions of kinase inhibition and enzyme-substrate relationships. PLOS Computational Biology, 20(5): e1012100.
11. Heid, E., et al. (2023). Chemprop: A Machine Learning Package for Chemical Property Prediction. Journal of Chemical Information and Modeling, 64(1): 9-17.
12. Morando, M.A., et al. (2016). Conformational Selection and Induced Fit Mechanisms in the Binding of an Anticancer Drug to the c-Src Kinase. Sci Rep, 6: 24439.
13. Roskoski Jr., R. (2016). Classification of small molecule protein kinase inhibitors based upon the structures of their drug-enzyme complexes. Pharmacological Research, 103: 26-48.
