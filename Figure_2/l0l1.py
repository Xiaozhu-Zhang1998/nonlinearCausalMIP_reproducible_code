from functions import *

# repeat the following for m in M = [i for i in range(30)]
# for each m, repeat for ll in range(20)
# example of m = 1 and ll = 1
m = 1
ll = 1


print("finished importing...")

# ======================================================================


# data info
data_name = inssmall
equalvar = False


edges_star = [(i-1, j-1) for (i,j) in data_name['edges']]
moral = [(i-1, j-1) for (i,j) in data_name['moral']]


if equalvar:
    name = "./dataset/" + data_name['name'] + "_equalvar/W_" + str(m) + ".csv"
else:
    name = "./dataset/" + data_name['name'] + "_diffvar/W_" + str(m) + ".csv"
W = np.loadtxt(name, delimiter=',')
W_train = W[0:500,:]


print("finished all setup!\n")


# ======================================================================

RS = np.zeros((0, 7))

# setup
n, p = W_train.shape
basis_setting = {'type': 'spline', 'n_knots': 2, 'degree': 2, 'include_intercept': False}
suff = gen_suff_stat(W_train, basis_setting)


Sigma_hat = suff["Sigma"]
R = suff["R"]
logLam = np.linspace(-9.2, -1.6, 20)


# l0 optimization ==========

t1 = time.time()
Mip_obj = Mips_DAG_diffvar(
    Sigma_hat, n, p, np.exp(logLam[ll]), R, moral, known_edges = [], partial_order = [], start_edges = [],
    tau = 0, verbose = 0, time_limit = p * 60)
t2 = time.time()

sd = np.tile(np.array([1, 0.5]), 15)[0:p]
diff = Mip_obj['varhat'] - sd
l2norm = np.sum(diff ** 2)

print(m, ll, logLam[ll], "perform:", edge_diff(edges_star, Mip_obj['g'], p), l2norm, t2 - t1, "\n\n\n")
print(Mip_obj['g'])
rs = np.array([m, ll, logLam[ll], "l0", edge_diff(edges_star, Mip_obj['g'], p), l2norm, t2 - t1])
RS = np.vstack([RS, rs])




# l1 optimization ==========

t1 = time.time()
Mip_obj = group_l1_opt(
    Sigma_hat, n, p, np.exp(logLam[ll]), R, moral, known_edges = [], partial_order = [], start_edges = [],
    tau = 0, verbose = 0, time_limit = p * 60)
t2 = time.time()

sd = np.tile(np.array([1, 0.5]), 15)[0:p]
diff = Mip_obj['varhat'] - sd
l2norm = np.sum(diff ** 2)

print(m, ll, logLam[ll], "perform:", edge_diff(edges_star, Mip_obj['g'], p), l2norm, t2 - t1, "\n\n\n")
print(Mip_obj['g'])
rs = np.array([m, ll, logLam[ll], "l1", edge_diff(edges_star, Mip_obj['g'], p), l2norm, t2 - t1])
RS = np.vstack([RS, rs])




# ======================================================================
name = "./results/" + str(m) + "_" + str(ll) + ".npy"
np.save(name, RS)



print("\n\n finished...")        
