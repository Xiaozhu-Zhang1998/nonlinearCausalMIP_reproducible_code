We will run 10 * 4 (data_structure * variance_scheme) combinations of experiment. For each one of them:

- Remember to specify the `data_name` and `equalvar` before running each file.

- Run the file `generate.py`. This generates 30 datasets.

- Put the true edge set as `data_name.txt` under the ./dataset folder.

- Run the file `R_baselines.R`. This generates 30 `.rds` files. For equal-variance scheme, bootstrapping results will be generated as well.

- Run the file `py_methods.py`. This generates 30 `.npy` files. For unequal-variance scheme, boostrapping results will be generated as well.

- Run the file `collect_R.R` to collect simulation results from R. This generates a file `R_outputs.csv`.

- Run the file `collect_py.py` to collect simulation results from python. This generates a file `Python_outputs.csv`.

---

Please save all files using the following directory hierarchy:

** variance scheme name (Equal, Unequal1, Unequal2, Unequal3)

*** graph name (dsep, asia, bowling, inssmall, rain, cloud, funnel, galaxy, insurance, factors)

**** R_outputs.csv

**** Python_outputs.csv

---

After finishing the above for all combinations, we generate figures by running the file `figure.R`, and generate tables by running the file `table.R`.