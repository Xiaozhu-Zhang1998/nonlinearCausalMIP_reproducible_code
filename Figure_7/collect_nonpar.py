from functions import *
print("finished importing...")

# ======================================================================

# data info
data_name = inssmall
equalvar = False


M = [i for i in range(30)]
RS = np.zeros((0, 7))
for m in M:
    for ll in range(15):
        name = "./results/" + str(m) + "_" + str(ll) + ".npy"
        rs = np.load(name)
        RS = np.vstack([RS, rs])
        print("finished", m, ll)

name = "./results/non_parametric.csv"


RS_df = pd.DataFrame(RS, columns = ["m", "l", "Rn", "l2", "diff", "perm", "time"])
RS_df.to_csv(name, index=False) 

print("finished...")
