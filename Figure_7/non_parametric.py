from functions import *

# repeat the following for m in M = [i for i in range(30)]
# for each m, repeat for ll in range(15)
# example of m = 1 and ll = 1
m = 1
ll = 1

print("finished importing...")

# ======================================================================


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

Kappa = [(i+1) / 30 for i in range(2,17)]
RS = np.zeros((0, 7))

# MIP using suff stat
n, p = W_train.shape
Rn = int(n ** Kappa[ll])
basis_setting = {'type': 'spline', 'n_knots': Rn, 'degree': 3, 'include_intercept': False}
suff = gen_suff_stat(W_train, basis_setting)


Sigma_hat = suff["Sigma"]
R = suff["R"]
lam = 0.01


if equalvar:
    MIP_fun = Mips_DAG_equalvar
else:
    MIP_fun = Mips_DAG_diffvar

t1 = time.time()
Mip_obj = MIP_fun(
            Sigma_hat, n, p, lam, R, moral, known_edges = [], partial_order = [], start_edges = [],
            tau = 0, verbose = 0, time_limit = 1800)
t2 = time.time()


sd = np.tile(np.array([1, 0.5]), 15)[0:p]
diff = Mip_obj['varhat'] - sd
l2norm = np.sum(diff ** 2)

# check wrong ordering
correct_order = True
for x,y in Mip_obj['g']:
    if x > y and (y, x) in edges_star:
        correct_order = False
        break


print(m, ll, Rn, "perform:", l2norm, edge_diff(edges_star, Mip_obj['g'], p), correct_order, t2 - t1)
print(Mip_obj['g'])
rs = np.array([m, ll, Rn, l2norm, edge_diff(edges_star, Mip_obj['g'], p), correct_order, t2 - t1])
RS = np.vstack([RS, rs])


# ======================================================================
name = "./results/" + str(m) + "_" + str(ll) + ".npy"
np.save(name, RS)



print("\n\n finished...")        