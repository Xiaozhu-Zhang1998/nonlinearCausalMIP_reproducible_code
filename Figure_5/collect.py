from functions import *
print("finished importing...")

# ======================================================================

# data info
data_name = inssmall
equalvar = False


M = [i for i in range(30)]
RS = np.zeros((0, 5))
for m in M:
    for ll in range(11):
        name = "./results/" + str(m) + "_" + str(ll) + ".npy"
        rs = np.load(name)
        RS = np.vstack([RS, rs])
        print("finished", m, ll)

name = "./results/early_stopping.csv"


RS_df = pd.DataFrame(RS, columns = ["m", "l", "tau", "diff", "time"])
RS_df.to_csv(name, index=False) 

print("finished...")
