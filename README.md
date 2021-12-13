# fine-grained-SC


# Introduction

This repository contains code and data for "[On the Assumptions of Synthetic Control Methods](https://arxiv.org/abs/2112.05671)".



# Data & code

1. Prop 99
- In 1989, California implemented a tobacco program that increased the tobacco tax by 25 cents.  We are interested in studying what the average smoking rate would be had the tobacco program never been implemented. 
- The data includes the average cigarette sales (measured in packs) of different states in American between 1970–2000. Here is the link to [data](https://github.com/claudiashi57/fine-grained-SC/blob/main/dat/prop99.csv)
- To reproduce figure 1 of the paper, run [src/prop99.ipynb](https://github.com/claudiashi57/fine-grained-SC/blob/main/src/prop99.ipynb)
- For a full data description, see [Abadie+ 2010](https://economics.mit.edu/files/11859)


2. Simulations
- We studied various implications of the paper using simulated data.
- Here is the [data generating process, src/dgp.py](https://github.com/claudiashi57/fine-grained-SC/blob/main/src/dgp.py)
- To reproduce figure 2, run [src/linear_v_nonlinear.ipynb](https://github.com/claudiashi57/fine-grained-SC/src/linear_v_nonlinear.ipynb)
- To reproduce figure 3, run [src/increase_s.ipynb](https://github.com/claudiashi57/fine-grained-SC/blob/main/src/increase_s.ipynb)
- to reproduce table 2, run [src/auxiliary_covariate.ipynb](https://github.com/claudiashi57/fine-grained-SC/blob/main/src/auxiliary_covariate.ipynb)
