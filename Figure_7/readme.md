We first run the file `generate.py` to generates 30 datasets.


For the nonlinear MIP, we implement the following in order:

- Run the file `non_parametric.py`. This generates 30 * 15 `.npy` files.

- Run the file `collect_nonpar.py` to collect simulation results. The generates a file `non_parametric.csv`.


For the linear MIP, we implement the following in order:

- Run the file `naive_MIP.py`. Then generates 30 `.npy` files.

- Run  the file `collect_naive.py` to collect simulation results. The generates a file `naive_MIP.csv`.



Finally, Run the file `figure.R` to generate the figure.

