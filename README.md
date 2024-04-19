# Efficient first principles based modeling via machine learning: from simple representations to high entropy materials

## Paper
This is the repo associated for our paper *Efficient first principles based modeling via machine learning: from simple representations to high entropy materials* ([publisher version](https://doi.org/10.1039/D4TA00982G), [arXiv version](https://arxiv.org/html/2403.15579v1)), which we create a large DFT dataset for HEMs and evaluate the in-distribution and out-of-distribution performance of machine learning models. 


## DFT dataset for high entropy alloys [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10854500.svg)](https://doi.org/10.5281/zenodo.10854500)

Our DFT dataset encompasses bcc and fcc structures composed of eight elements and overs all possible 2- to 7-component alloy systems formed by them. 
The dataset used in the paper is publicly available on [Zenodo](https://doi.org/10.5281/zenodo.10854500), which includes initial and final structures, formation energies, atomic magnetic moments and charges among other attributes. 

*Note: The trajectory data (energies and forces for structures during the DFT relaxations) is not published with this paper; it will be released later with a work on machine learning force fields for HEMs.*


### Table: Numbers of alloy systems and structures.
| No. components             | 2    | 3     | 4     | 5     | 6    | 7    | Total |
|----------------------------|------|-------|-------|-------|------|------|-------|
| Alloy systems              | 28   | 56    | 70    | 56    | 28   | 8    | 246   |
| Ordered (2-8 atoms)        | 4975 | 22098 | 29494 | 6157  | 3132 | 3719 | 69575 |
| SQS (27, 64, or 128 atoms) | 715  | 3302  | 3542  | 4718  | 1183 | 762  | 14222 |
| Ordered+SQS                | 5690 | 25400 | 33036 | 10875 | 4315 | 4481 | 83797 |


### Number of structures as a function of a given constituent element. 
The legend indicates the number of components.
<p align="left" width="100%">
    <img src="figs/counts_vs_elements.png" alt="image" width="40%" height="auto">
</p>

## Generalization performance of machine learning models 
The data on [Zenodo](https://doi.org/10.5281/zenodo.10854500) provide the Matminer features of initial and final structures and a demo script to train tree-based models. The results in the paper can be readily reproduced by adapting the demo script for different train-test splits. The `codes` folder provides the scripts that we used in the paper.

### Generalization performance from small to large structures.
<p align="center" width="100%">
    <img src="figs/fig2.png" alt="image" width="100%" height="auto">
    <br>
    <em>(a) Normalized error obtained by training on structures with ≤ N atoms and evaluating on structures with > N atoms. (b) ALIGNN prediction on SQSs with > 27 atoms, obtained by training on structures with ≤ 4 atoms. (c) Parity plot of the ALIGNN prediction on SQSs with > 27 atoms, obtained by training on structures with ≤ 8 atoms.</em>
</p>
&nbsp;&nbsp; 

### Generalization performance from low-order to high-order systems.
<p align="center" width="100%">
    <img src="figs/fig3.png" alt="image" width="100%" height="auto">
    <br>
    <em>(a) Normalized error obtained by training on structures with ≤ N elements and evaluating on structures with >N elements. (b) Parity plot of the ALIGNN prediction on structures with ≥ 3 elements, obtained by training on binary structures. (c) Parity plot of the ALIGNN prediction on structures with ≥ 4 elements, obtained by training on binary and ternary structures.</em>
</p>
&nbsp;&nbsp; 

### Generalization performance from (near-)equimolar to non-equimolar structures.
<p align="center" width="100%">
    <img src="figs/fig4.png" alt="image" width="100%" height="auto">
    <br>
    <em>(a) Normalized error obtained by training on structures with maxΔc below a given threshold and evaluating on the rest. (b) Predictions on non-equimolar structures (maxΔc>0) by the ALIGNN model trained on equimolar structures (maxΔc=0). (c) Predictions on structures with relatively strong deviation from equimolar composition (maxΔc > 0.2) by the ALIGNN model trained on structures with relatively weak deviation from equimolar composition (maxΔc ≤ 2). maxΔc is defined as the maximum concentration difference between any two elements in a structure.</em>
</p>
&nbsp;&nbsp; 

### Effects of dataset size and use of unrelaxed vs. relaxed structures
<p align="center" width="100%">
    <img src="figs/fig5.png" alt="image" width="100%" height="auto">
</p>
&nbsp;&nbsp; 

### Overview of model performance on different generalization tasks
<p align="center" width="100%">
    <img src="figs/table2.png" alt="image" width="100%" height="auto">
</p>


