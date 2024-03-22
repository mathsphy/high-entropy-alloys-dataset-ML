# high-entropy-alloys-dataset-ML

## Dataset description [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10854500.svg)](https://doi.org/10.5281/zenodo.10854500)

Our DFT dataset encompasses bcc and fcc structures composed of eight elements and overs all possible 2- to 7-component alloy systems formed by them. 
The dataset in the csv format is publicly available on [Zenodo](https://doi.org/10.5281/zenodo.10854500), which includes the relaxed formation energies, the initial and final structures, and the Matminer features of the final structures, among other attributes.

**Table: Numbers of alloy systems and structures.**
| No. components             | 2    | 3     | 4     | 5     | 6    | 7    | Total |
|----------------------------|------|-------|-------|-------|------|------|-------|
| Alloy systems              | 28   | 56    | 70    | 56    | 28   | 8    | 246   |
| Ordered (2-8 atoms)        | 4975 | 22098 | 29494 | 6157  | 3132 | 3719 | 69575 |
| SQS (27, 64, or 128 atoms) | 715  | 3302  | 3542  | 4718  | 1183 | 762  | 14222 |
| Ordered+SQS                | 5690 | 25400 | 33036 | 10875 | 4315 | 4481 | 83797 |


<p align="center" width="100%">
    <img src="figs/counts_vs_elements.png" alt="image" width="60%" height="auto">
    <br>
    <em><strong>Figure: Number of structures as a function of a given constituent element. The legend indicates the number of components.</strong> </em>
</p>

## Featurization and machine learning models 
The data on [Zenodo](https://doi.org/10.5281/zenodo.10854500)




