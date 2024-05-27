# Lantern Pharma Drug Response Prediction Challenge
Major Goal: Development of interpretable machine learning models to predict response to drugs and drug combinations. Model development can lead to clinically relevant information on drug mechanisms and which patients should receive drugs or drug combinations at an individual level.

## Introduction
Many cancer drugs have shown clinical benefit to various patients, but it is well-known that individual tumor response to effective drugs can vary. While some drug response variation may simply be due to cancer type (e.g., lung vs skin cancer), most reasons for individual variation to specific drugs remain unclear, and predicting individual variation to drug response is a complex challenge. In some cases drug response is due to resistance mechanisms caused by genetic mutations, or changes to the expression of genes in tumor cells.

A research group has examined 4 FDA-approved drugs with variable efficacy across a panel of 9 different cancer types and wants to determine molecular features that predict drug response. Popular assays of molecular characteristics were performed across 48 cellular models of these 9 cancer types. The cellular models are known as “cell lines”, which originate from tumor samples that grow in lab environments indefinitely. Based on information of the cancer type and given molecular features, development of predictive models that predict response to individual response or combinations will be the goal of this challenge, which will be scored by evaluation of prediction accuracy on test data that will be provided after submission. To develop models, participants will be provided with a drug response. Drug response was measured by determining the inhibition of cancer growth or cell death after transient drug exposure. In the data provided, values of 1 represent samples that were considered responsive (i.e., effective in killing cancer cells) to a drug or drug combination, and 0 represents a lack of response (representing cancers that are resistant to the given therapy). Matching molecular features are provided for the cell line samples where drugs were tested.

**Cancer type labels**:
Information of the broad tissue origin is labeled according to the following 9 classifications: "Breast" "BrainCNS" "Bowel" "Lung" "Blood" "Skin" "Ovary" "Prostate" "Kidney"

**Mutation data**:
Mutations are changes to the genetic sequence in the genome. In this challenge, mutations to the protein coding gene sequences have been taken for each gene of every sample using whole-genome sequencing (WGS). Mutations are considered to play a defining causal role in cancer development and progression. In fact, anti-cancer drugs are commonly designed to be given to patients with specific mutations (known as “targeted therapies”). Despite capability to target key mutations with drugs, presence of specific mutations often fails to predict cancer response to targeted therapies.
After “high-throughput sequencing” of the entire genome (containing >20,000 protein coding genes), mutations which were considered likely to impact protein function were identified in each sample. In the data provided, these mutations are represented in a binary format, with 1 representing a mutation, and 0 representing a non-mutated (normal) gene. These features are labeled as “mut_” followed by the gene’s ID.

**RNA data**:
Ribonucleic acid (RNA) is the genetic product of genes that are read and transcribed from DNA in the nucleus. RNA is later converted into protein, which is considered the functional output of our genetic code. Proteins will perform numerous proteins and their abundance in the cell is mostly determined by the RNA levels of corresponding gene sequences. RNA levels (sometimes referred to as mRNA or transcript abundance) have innumerable associations with cellular functions, including tumor response to various drugs. RNA is most frequently measured for every gene by RNA sequencing (RNA-seq) assays.
In data provided, RNA-seq data has been processed in a standardized format. After performing a standard length normalization that converts counts of genes read in sequencing to Fragments Per Kilobase Million (FPKM), it has been provided in the format of log2(FPKM+1). These features are labeled as “rna_” followed by the gene’s ID.

## Scoring

Submissions will be scored automatically based on accuracy (F1_score) of model predictions with a set of test data. Additionally, short answer questions will be evaluated by judges at Lantern for 20% weight (only top performing scores will be evaluated). It is required to submit specific files described below into a Results folder and for scripts used for model generation to be in participant's Code Ocean capsule. 

## Challenge Deliverables

1. Two separate tables will be submitted to both describe each model used and give their prediction results on the test set. For the submission, only a single chosen model should be described for each of the 6 drug or drug/combinations to be modeled. If more than one model is available for different drug or drug/combination predictions, choose the model believed to be the best. Details of expected output for table submissions are described below.
2. Assessment of the most important feature of a model’s predictions is required.
