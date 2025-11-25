from functions import *
print("finished importing...")

# ======================================================================

# repeat the following for n0 in [300, 400, ..., 1500, 1600, 2000, 2500]
# example of n0 = 2500
n0 = 2500

M = [i for i in range(30)]
RS = np.zeros((0, 6))
for m in M:
    name = "./suff_stat/" + str(m) + ".npy"
    rs = np.load(name)
    RS = np.vstack([RS, rs])


name = "./suff_stat_output_n" + str(n0) + ".csv"

RS_df = pd.DataFrame(RS, columns = ["m", "method", "time_obj", "time_bigM", "time_opt", "time_total"])
RS_df.to_csv(name, index=False) 

print("finished...")