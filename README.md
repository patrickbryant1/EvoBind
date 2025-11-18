

# EvoBind
<img align="right" src="./EB_logo.png">

**In silico directed evolution of peptide binders**
\
\
EvoBind (v2) designs novel peptide binders based **only on a protein target sequence**. It is not necessary to specify any target residues within the protein sequence or the length of the binder (although this is possible). **Cyclic binder** design is also possible.
\
\
[Read more here](https://www.nature.com/articles/s42004-025-01601-3)
\
\
EvoBind2 accounts for adaptation of the receptor interface structure to the peptide being designed during optimisation: **sequence and structure is generated simultaneously**. This consideration of flexibility is crucial for binding. EvoBind is the first protocol that only relies on a protein sequence to design a binder with experimentally verified cyclic design capacity.


<p align="center">
  <img alt="Linear" src="./linear.gif" width="45%">
&nbsp; &nbsp; &nbsp; &nbsp;
  <img alt="Cyclic" src="./cyclic.gif" width="45%">
</p>

Receptor in green and peptide in blue.

# If you like EvoBind - please star the repo!

# Table of Contents
1.  [EvoBind](#evobind)
2.  [LICENSE](#license)
3.  [Colab](#colab)
4.  [Computational requirements](#computational-requirements)
5.  [Setup](#setup)
6.  [Design binders](#design-binders)
    * [Cyclic design](#cyclic-design)
    * [Adversarial evaluation with AlphaFold-multimer](#adversarial-evaluation-with-alphaFold-multimer)
8. [Citation](#citation)
9. [Examples of studies with EvoBind](#examples-of-studies-with-evobind)
10. [The EvoBind ecosystem](#the-evobind-ecosystem)


# LICENSE
EvoBind2 is based on AlphaFold2, which is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).  \
The AlphaFold2 parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode) and have not been modified. \
The design protocol EvoBind2 is made available under the terms of the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
\
**You may not use these files except in compliance with the licenses.**

# Colab
It is possible to run EvoBind2 online in the [Google colab here](https://colab.research.google.com/github/patrickbryant1/EvoBind/blob/master/EvoBind.ipynb)

# Computational requirements
Before beginning the process of setting up this pipeline on your local system, make sure you have adequate computational resources. Make sure you have an **available GPU** as this will speed up the prediction process substantially compared to using a CPU. EvoBind2 assumes you have NVIDIA GPUs on your system, readily available. A Linux-based system is assumed.

# Setup
To setup this pipeline, clone this github repository:
```
git clone https://github.com/patrickbryant1/EvoBind.git
```
\
Then do
```
bash setup.sh
```
This script fetches the [AlphaFold2 parameters](https://storage.googleapis.com/alphafold/alphafold_params_2021-07-14.tar), installs a conda env and downloads [uniclust30_2018_08](http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz) which is used to generate the receptor MSA.

# Design binders
To design binders the following needs to be specified: \
**Receptor fasta sequence** \
Optional arguments:
\
**Peptide length** - default=10 \
**Target residues within the raceptor sequence** - default=all

## Cyclic design
If you want to design a cyclic peptide, add the flag --cyclic_offset=1 in the design script when calling mc_design.py. Based on [cyclic offset](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9980166/).

A test case is provided in **design_local.sh**. \
This script can be run by simply doing:
```
bash design_local.sh
```

## Adversarial evaluation with AlphaFold-multimer
See **src/AFM_eval** for instructions

# Citation
If you use EvoBind in your research, please cite

[Li, Q., Vlachos, E.N. & Bryant, P. Design of linear and cyclic peptide binders from protein sequence information. Commun Chem 8, 211 (2025). https://doi.org/10.1038/s42004-025-01601-3](https://www.nature.com/articles/s42004-025-01601-3)

# Examples of studies with EvoBind
1. [Daumiller D*, Giammarino F*, Li Q, Sonnerborg A, Cena-Diez R, Bryant P. Single-Shot Design of a Cyclic Peptide Inhibitor of HIV-1 Membrane Fusion with EvoBind. bioRxiv. 2025. p. 2025.04.30.651413. doi:10.1101/2025.04.30.651413](https://www.biorxiv.org/content/10.1101/2025.04.30.651413v1)
2. [Li Q, Wiita E, Helleday T, Bryant P. Blind De Novo Design of Dual Cyclic Peptide Agonists Targeting GCGR and GLP1R. bioRxiv. 2025. p. 2025.06.06.658268. doi: https://doi.org/10.1101/2025.06.06.658268](https://www.biorxiv.org/content/10.1101/2025.06.06.658268v1)

# The EvoBind ecosystem
[RareFold](https://github.com/patrickbryant1/RareFold) - prediction & design with noncanonical amino acids \
[RareFoldGPCR](https://github.com/patrickbryant1/RareFoldGPCR) - GPCR agonist design with noncanonical amino acids
