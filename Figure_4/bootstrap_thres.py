from functions import *

# for tau_s = tau_p, set flag_equal_thres = True; otherwise set flag_equal_thres = False
# repeat the following for m in M = [i for i in range(30)]
# example of m = 1 and flag_equal_thres = True

m = 1
flag_equal_thres = True


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
W_train = W[0:1000,:]
n,p = W_train.shape


print("finished all setup!\n")


# ======================================================================

RS = np.zeros((0, 5))

if equalvar:
    name_RS = "./dataset/" + data_name['name'] + "_equalvar/performThres_" + str(m) + ".npy"
    name_prob = './dataset/' + data_name['name'] + '_equalvar/prob_adj_CAM_' + str(m) + '.csv'
else:
    name_RS = "./dataset/" + data_name['name'] + "_diffvar/performThres_" + str(m) + ".npy"
    name_prob = './dataset/' + data_name['name'] + '_diffvar/prob_adj_CAM_' + str(m) + '.csv'


# do bootstrap using CAM
print("started bootstrapping CAM...")
B = 20
Adj = np.zeros((p,p))
for b in range(B):
    idx = np.random.choice(list(range(W_train.shape[0])), W_train.shape[0], replace = True)
    W_idx = W_train[idx,:]
    camobj = CAM(score='nonlinear', cutoff=0.001, variablesel=True, pruning=True)
    output = camobj.predict(pd.DataFrame(W_idx))
    Adj = Adj + edge2adj(list(output.edges), p)
    print("finished", b)
prob_adj = Adj / B
print("finished bootstrapping CAM...", "\n")



Thres = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
lam = 0.01


for kappa_par in Thres:
    if flag_equal_thres:
        kappa_stab = kappa_par
    else:
        kappa_stab = 1
    

    # find partial order set
    par_edges = list()
    par_vals = np.array([])
    for i,j in moral:
        if prob_adj[i,j] >= kappa_par and prob_adj[j,i] <= 1 - kappa_par:
            par_edges.append((i,j))
            par_vals = np.append(par_vals, prob_adj[i,j])
        if prob_adj[j,i] >= kappa_par and prob_adj[i,j] <= 1 - kappa_par:
            par_edges.append((j,i))
            par_vals = np.append(par_vals, prob_adj[j,i])
    sorted_idx = np.argsort(-par_vals)
    par_vals = par_vals[sorted_idx]
    par_edges = [par_edges[i] for i in sorted_idx]
    i = 1
    while i < len(par_vals):
        if isCyclic(p, par_edges[0:i]):
            par_vals = np.delete(par_vals, i)
            par_edges.pop(i)
        else:
            i += 1
    print("found partial order set")


    # find stable set
    stab_edges = list()
    stab_vals = np.array([])
    for i in range(p):
        for j in range(p):
            if prob_adj[i,j] >= kappa_stab and (i,j) in par_edges:
                stab_edges.append((i,j))
                stab_vals = np.append(stab_vals, prob_adj[i,j])
    sorted_idx = np.argsort(-stab_vals)
    stab_vals = stab_vals[sorted_idx]
    stab_edges = [stab_edges[i] for i in sorted_idx]
    i = 1
    while i < len(stab_vals):
        if isCyclic(p, stab_edges[0:i]):
            stab_vals = np.delete(stab_vals, i)
            stab_edges.pop(i)
        else:
            i += 1
    print("found stable set")


    # run the MIP
    if equalvar:
        MIP_fun = Mips_DAG_equalvar
    else:
        MIP_fun = Mips_DAG_diffvar
    suff = gen_suff_stat(W_train, basis_setting)
    Sigma_hat = suff["Sigma"]
    R = suff["R"]

    try:
        t1 = time.time()
        Mip_obj = MIP_fun(
            Sigma_hat, n, p, lam, R, moral, known_edges = stab_edges, partial_order = par_edges, start_edges = [],
            tau = 0, verbose = 0, time_limit = p * 60)
        t2 = time.time()
    except Exception:
        continue

    print("kappa_stab:", kappa_stab, "kappa_par:", kappa_par, "perfom:", edge_diff(edges_star, Mip_obj['g'], p), "time:", t2 - t1, '\n' )

    rs = np.array([m, kappa_stab, kappa_par, edge_diff(edges_star, Mip_obj['g'], p), t2 - t1 ])
    RS = np.vstack([RS, rs])


np.save(name_RS, RS)


print("\n\n finished...")        
