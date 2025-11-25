from functions import *

# specify the data structure and variance scheme
# repeat the following for m in M = [i for i in range(30)]
# example of m = 1, data_name = inssmall, equal-variance, and mu0 = 2

data_name = inssmall
equalvar = False
mu0 = 2
m = 1

print("finished importing...")



# true functions
def func(x):
    return (np.sin(x) + np.cos(x)) / 2 

edges_star = [(i-1, j-1) for (i,j) in data_name['edges']]

n = 5000
p = data_name['p']



print("finished all setup!\n")

rng = np.random.default_rng()
# generate data
if equalvar:
    W = gen_data(edges_star, n = n, p = p, func = func, sd = np.array([0.5]))
    name = "./dataset/" + data_name['name'] + "_equalvar/W_" + str(m) + ".csv"
else:
    sd = rng.beta(a=1, b=mu0, size=p) * 0.5 + 0.5
    W = gen_data(edges_star, n = n, p = p, func = func, sd = sd)
    name = "./dataset/" + data_name['name'] + "_diffvar/W_" + str(m) + ".csv"
    
np.savetxt(name, W, delimiter=',')

        
print("finished m=", m, "...")        
