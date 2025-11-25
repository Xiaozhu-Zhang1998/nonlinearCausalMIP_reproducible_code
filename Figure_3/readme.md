First, Run the file `generate.py`. This generates 30 datasets.


For each sample size $n_0$, we implement the following in order:

- Specify `n0` and run the file `suff_stat.py` for each `m`. This generates 30 `.npy` files.

- Specify `n0` and run the file `collect.py`. This generates a file `suff_stat_output_n" + str(n0) + ".csv`.


After finishing the above for all sample sizes, we generate figures by running the file `figure.R`.
