
from functions import *


m = 0
data_name = causal_chamber
equalvar = True


print("finished importing...")


# ======================================================================


edges_star = [(i-1, j-1) for (i,j) in data_name['edges']]
moral = [(i-1, j-1) for (i,j) in data_name['moral']]

basis_setting = {'type': 'spline', 'n_knots': 2, 'degree': 2, 'include_intercept': False}
name = "W_0.csv"
W = np.loadtxt(name, delimiter=',')
W_train = W[0:3000,:]
n,p = W_train.shape


print("finished all setup!\n")


# ======================================================================

RS = np.zeros((0, 5))

if equalvar:
    name_CAM = "./dataset/" + data_name['name'] + "_equalvar/CAM_" + str(m) + ".pkl"
    name_pre = "./dataset/" + data_name['name'] + "_equalvar/pre_" + str(m) + ".pkl"
else:
    name_CAM = "./dataset/" + data_name['name'] + "_diffvar/CAM_" + str(m) + ".pkl"
    name_pre = "./dataset/" + data_name['name'] + "_diffvar/pre_" + str(m) + ".pkl"



# ======================================================================



# CCDr
t1 = time.time()
ccdrobj = CCDr()
output = ccdrobj.predict(pd.DataFrame(W_train))
t2 = time.time()
rs = np.array([m, p, "CCDr", edge_diff(edges_star, list(output.edges), p), t2 - t1])
RS = np.vstack([RS, rs])
print("For m:", m, "Finished CCDr:", edge_diff(edges_star, list(output.edges), p), "\n")


# ======================================================================

# CAM cv
Cutoff = np.exp(np.linspace( np.log(0.1), np.log(0.00001), 10))
CAM_cv_obj = CAM_cv(W_train, edges_star, basis_setting, Cutoff = Cutoff)
rs = np.array([m, p, "CAM", edge_diff(edges_star, CAM_cv_obj['edges'], p), CAM_cv_obj['time']])
RS = np.vstack([RS, rs])
print("For m:", m, "Finished CAM cv:", edge_diff(edges_star, CAM_cv_obj['edges'], p), "\n")

with open(name_CAM, "wb") as f: # 'wb' for write binary
    pickle.dump(CAM_cv_obj, f)


# ======================================================================


reverse = False
# prep
if equalvar:
    name = "./dataset/" + data_name['name'] + "_equalvar/s0_NPVAR_" + str(m) + ".txt"
    with open(name, 'r') as file:
        s0_max = float(file.read())
else:
    s0_max = len(CAM_cv_obj['edges']) 

alphas = np.exp(np.linspace( np.log(0.1), np.log(0.0001), 20))
MIP_pre_process_obj = MIP_pre_process(
    W_train, data_name, m, equalvar, basis_setting,
    moral = None, alphas = alphas, s0_max = s0_max * 2,        # for moral
    cutoff = CAM_cv_obj['cutoff'], Loss_CAM = CAM_cv_obj['loss_val'], B = 20, 
    kappa_par = 0.95, kappa_stab = 1,  # for bootstrapping
    reverse = reverse
)

c0 = np.log(p) / n * p / s0_max
if MIP_pre_process_obj['CAM_bootstrap_time'] is not None:
    rs = np.array([m, p, "Bootstrapping", np.nan, MIP_pre_process_obj['CAM_bootstrap_time']])
    RS = np.vstack([RS, rs])

with open(name_pre, "wb") as f: # 'wb' for write binary
    pickle.dump(MIP_pre_process_obj, f)

# ======================================================================


# MIP cv (estimated superset)
superset = MIP_pre_process_obj['moral']
print("superset:", superset)
print("superset size:", len(superset))
edges_unselected = [pair for pair in edges_star if pair not in superset]
print("edges not included:", len(edges_unselected), ":", edges_unselected )

Lam = np.linspace(c0 * 10, c0 * 50, 10)

MIP_cv_obj = MIP_cv(
    W_train, edges_star, basis_setting, Lam = Lam,
    moral = MIP_pre_process_obj['moral'], stab_edges = MIP_pre_process_obj['stab_edges'], 
    par_edges = MIP_pre_process_obj['par_edges'], equalvar = equalvar, tau_const = 0.1, time_limit = p * 60
)
print("For m:", m, "Finished MIP cv: knots", basis_setting['n_knots'], "\n\n")


rs = np.array([m, p, "MIP (super)", edge_diff(edges_star, MIP_cv_obj['edges'], p), MIP_cv_obj['time']])
RS = np.vstack([RS, rs])



# ======================================================================


# NoTears
Lam = np.exp(np.linspace( np.log(10), np.log(0.001), 10))
notears_cv_obj = notears_cv(W_train, edges_star, basis_setting, Lam = Lam)
rs = np.array([m, p, "NoTears", edge_diff(edges_star, notears_cv_obj['edges'], p), notears_cv_obj['time']])
RS = np.vstack([RS, rs])
print("For m:", m, "Finished NoTears:", edge_diff(edges_star, notears_cv_obj['edges'], p), "\n")



# ======================================================================



RS_df = pd.DataFrame(RS, columns = ["m", "p", "method", "diff", "time"])
RS_df.to_csv("Python_outputs.csv", index=False) 



print("\n\n finished...")        
