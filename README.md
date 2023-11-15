# Code and Data for Semi-nonparametric Varying Coefficients Models for Imaging Genetics
This is the ReadMe document for running the simulation and real data analysis presented in the paper.

## Data
Data used in the paper.
- final_clinical_m12_used.csv: it contains the clinical information.
 - the first column is the ID for each subject
 - the second and the third columns denote Gender and Handedness
 - the fourth to the sixth columns denote the Marital Status (if Widowed, the corresponding element of the fourth column would be 1; if Divorced, the corresponding element of the fifth column would be 1; if Never married, the corresponding element of the sixth column would be 1; If fourth to the sixth are all zero, it corresponds to Married)
 - the seventh to the ninth columns denote Education length, Retirement status and Age respectively.
 -  the last 5 columns contain the leading 5 principal components of all the genetic data

- Hippocampus_lqd_m_20left.txt: the lqd data for the left hippocampus, the first row is the grid.
- Gene_for_left: the folder contain the top 10 significant genetic data and label information of each block for the left, respectively.
  - For example: Gene_chr1_blcok_10.csv contains the genetic data in block10 of chromosome 1.
  - Label_chr1_blcok_10.csv contains the information of SNPs in this block.
  - The first column represents the number of chromosome, the second is the name of the snp, the "location" represents the position of the SNP. The "position" represents the position of the SNP within chromosome 1 (no use).

## Code
Python code for implementing the proposed method.
- SVC.py: all the estimation and testing functions
- RunSVC.py: the simulation settings and the main function to run the code
- Demo code.py: demo code for the simulation and real data
