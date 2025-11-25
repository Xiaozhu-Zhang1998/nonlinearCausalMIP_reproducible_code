For each function $h(x)$, we implement the following in order:

- Specify `func_name` and `h` in the file `CAM_NPVAR.R`.

- Run the file `CAM_NPVAR.R` for each `n` and independent trial. This generates 20 * 100 datasets and a file `func_name.csv`.

- Run the file `MIP.ipynb`. This generates a file `func_name_python.csv`.


After finishing the above for all functions $h(x)$, we generate figures by running the file `figure.R`.