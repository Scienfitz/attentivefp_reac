# SMR Deep Learning using Attentive Fingerprints #

This repo implements the workflows to use **Attentive Fingerprint** with small molecule data.

It includes workflows for:
* Training
* Cross-Validation with SMR specific CV modes
* Baseline sklearn model
* Batch inference
* Preliminary REST-API

This repo uses Pytorch as the DL library and is build around the [DGL library](https://github.com/awslabs/dgl-lifesci).

AttentiveFP were introduced in the publication [Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism](https://www.ncbi.nlm.nih.gov/pubmed/31408336).