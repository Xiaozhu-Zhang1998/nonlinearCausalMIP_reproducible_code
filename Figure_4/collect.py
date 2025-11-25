from functions import *
print("finished importing...")

# ======================================================================

# for tau_s = tau_p, set flag_equal_thres = True; otherwise set flag_equal_thres = False
# example of flag_equal_thres = True

flag_equal_thres = True

# data info
data_name = insurance
equalvar = False


M = [i for i in range(30)]
RS = np.zeros((0, 5))
for m in M:
    if equalvar:
        name = "./dataset/" + data_name['name'] + "_equalvar/performThres_" + str(m) + ".npy"
    else:
        name = "./dataset/" + data_name['name'] + "_diffvar/performThres_" + str(m) + ".npy"
    rs = np.load(name)
    RS = np.vstack([RS, rs])


if flag_equal_thres:
    name = "performThres_outputs_varied_stab.csv"
else:
    name = "performThres_outputs_fixed_stab.csv"

RS_df = pd.DataFrame(RS, columns = ["m", "thres_stab", "thres_par", "diff", "time"])
RS_df.to_csv(name, index=False) 

print("finished...")
