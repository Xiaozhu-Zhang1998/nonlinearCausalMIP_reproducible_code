# specify the data_name and variance scheme
# example of structure `dsep` and equal-variance


from functions import *
print("finished importing...")

# ======================================================================

# data info
data_name = dsep
equalvar = True


M = [i for i in range(30)]
RS = np.zeros((0, 5))
for m in M:
    if equalvar:
        name = "./dataset/" + data_name['name'] + "_equalvar/RS_" + str(m) + ".npy"
    else:
        name = "./dataset/" + data_name['name'] + "_diffvar/RS_" + str(m) + ".npy"
    rs = np.load(name)
    RS = np.vstack([RS, rs])


if equalvar:
    name = "./dataset/" + data_name['name'] + "_equalvar/Python_outputs" + ".csv"
else:
    name = "./dataset/" + data_name['name'] + "_diffvar/Python_outputs" + ".csv"

RS_df = pd.DataFrame(RS, columns = ["m", "p", "method", "diff", "time"])
RS_df.to_csv(name, index=False) 

print("finished...")
