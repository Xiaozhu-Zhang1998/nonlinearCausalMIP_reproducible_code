First, run the file `generate.py`. This generates 30 datasets.


For tau_s = tau_p, set `flag_equal_thres = True`; otherwise set `flag_equal_thres = False`. For each `flag_equal_thres` value,

- Run the file `bootstrap_thres.py`. This generates 30 `.npy` files.

- Run the file `collect.py`. This generates a file `performThres_outputs_varied_stab.csv` or `performThres_outputs_fixed_stab.csv`


After finishing the above for both `flag_equal_thres` values, we generate figures by running the file `figure.R`.