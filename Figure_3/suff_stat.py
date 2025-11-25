from functions import *

# repeat the following for n0 in [300, 400, ..., 1500, 1600, 2000, 2500]
# for each n0, repeat for m in M = [i for i in range(30)]
# example of n0 = 2500 and m = 1
n0 = 2500
m = 1


print("finished importing...")

# ======================================================================


# data info
data_name = inssmall
equalvar = False


edges_star = [(i-1, j-1) for (i,j) in data_name['edges']]
moral = [(i-1, j-1) for (i,j) in data_name['moral']]


basis_setting = {'type': 'spline', 'n_knots': 2, 'degree': 2, 'include_intercept': False}

if equalvar:
    name = "./dataset/" + data_name['name'] + "_equalvar/W_" + str(m) + ".csv"
else:
    name = "./dataset/" + data_name['name'] + "_diffvar/W_" + str(m) + ".csv"
W = np.loadtxt(name, delimiter=',')
W_train = W[0:n0,:]


print("finished all setup!\n")


# ======================================================================

RS = np.zeros((0, 6))

# MIP using suff stat
n, p = W_train.shape
suff = gen_suff_stat(W_train, basis_setting)
Sigma_hat = suff["Sigma"]
R = suff["R"]
lam = 0.5

t1 = time.time()
Mip_suff_obj = Mips_suff(Sigma_hat, n, p, lam, R, moral, time_limit = 3000, known_edges = [], partial_order = [], start_edges = [], tau = 0, verbose = 0)
t2 = time.time()

print(m, "suff_stat", Mip_suff_obj['time_obj'], Mip_suff_obj['time_bigM'], Mip_suff_obj['time_opt'], t2 - t1)
rs = np.array([m, "suff_stat", Mip_suff_obj['time_obj'], Mip_suff_obj['time_bigM'], Mip_suff_obj['time_opt'], t2 - t1])
RS = np.vstack([RS, rs])


# MIP default
basis_mat_obj = gen_basis_matrix(W_train, basis_setting)
basis_mat = basis_mat_obj["basis_mat"]
R = basis_mat_obj["R"]
X = W_train

t3 = time.time()
Mip_default_obj = Mips_default(X, basis_mat, n, p, lam, R, moral, time_limit = 3000, known_edges = [], partial_order = [], start_edges = [], tau = 0, verbose = 0)
t4 = time.time()

print(m, "default", Mip_default_obj['time_obj'], Mip_default_obj['time_bigM'], Mip_default_obj['time_opt'], t4 - t3)
rs = np.array([m, "default", Mip_default_obj['time_obj'], Mip_default_obj['time_bigM'], Mip_default_obj['time_opt'], t4 - t3])
RS = np.vstack([RS, rs])


# ======================================================================
name = "./suff_stat/" + str(m) + ".npy"
np.save(name, RS)



print("\n\n finished...")        