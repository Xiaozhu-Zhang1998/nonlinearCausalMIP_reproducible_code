The dataset `W_0.csv` is the post-processing data. See the file `processing.ipynb` for details.


We implement the following in order:

- Run the file `R_baselines.R`. This generates a file `R_outputs.csv`. Bootstrapping results based on NPVAR will be generated as well.

- Run the file `py_methods.py`. This generates a file `Python_outputs.csv`. Boostrapping results based on CAM will be generated as well.

- Run the file `result.R` to view the final results.
